"""
Module: config/thresholds.py
Purpose: All signal thresholds, risk parameters, and quality filter constants.
         Every magic number in the system lives here. No exceptions.
Phase: All
Last modified: 2026-06-11
"""

# ---------------------------------------------------------------------------
# Dataset Quality Filters (§5.4)
# ---------------------------------------------------------------------------
MIN_POST_COUNT_1D: int = 3          # exclude rows with fewer posts in 24h window
MIN_AVG_SENTIMENT_CONFIDENCE: float = 0.5  # mean FinBERT confidence across window
WINSORIZE_SIGMA: float = 5.0        # clip return targets at ±5 std deviations

# ---------------------------------------------------------------------------
# Signal Generation (§8)
# ---------------------------------------------------------------------------
SIGNAL_BUY_MIN_PRED_5D: float = 0.03        # predicted 5d return must exceed this for BUY
SIGNAL_AVOID_MAX_PRED_5D: float = -0.03     # predicted 5d return must be below this for AVOID
SIGNAL_CONFIDENCE_THRESHOLD: float = 0.70   # minimum confidence to generate BUY or AVOID
SIGNAL_MIN_RVOL: float = 1.2                # relative volume gate for BUY
SIGNAL_MIN_SENTIMENT_ACCEL: float = 0.0     # sentiment acceleration must be > 0 for BUY
MIN_CONFIDENCE_FOR_ANY_SIGNAL: float = 0.50 # below this → HOLD regardless of prediction

# ---------------------------------------------------------------------------
# Signal Ranking (§8)
# ---------------------------------------------------------------------------
RANK_WEIGHT_RETURN: float = 0.5
RANK_WEIGHT_CONFIDENCE: float = 0.3
RANK_WEIGHT_ACCEL: float = 0.2
MAX_BUY_SIGNALS: int = 4   # top-N to forward to portfolio engine

# ---------------------------------------------------------------------------
# Portfolio Hard Limits (§9.1)
# ---------------------------------------------------------------------------
MAX_POSITIONS: int = 4
MAX_POSITION_PCT: float = 0.25         # max 25% of portfolio per trade
MIN_CASH_RESERVE_PCT: float = 0.20     # always keep ≥20% cash
MAX_TOTAL_EXPOSURE_PCT: float = 0.75   # max 75% invested at any time
MAX_SECTOR_CONCENTRATION_PCT: float = 0.40  # max 40% in one sector
DAILY_LOSS_LIMIT_PCT: float = -0.03    # -3% daily → halt new trades
WEEKLY_LOSS_LIMIT_PCT: float = -0.07   # -7% weekly → pause system

# ---------------------------------------------------------------------------
# Position Sizing (§9.2)
# ---------------------------------------------------------------------------
# multiplier = POSITION_SIZE_BASE + confidence  →  range [1.0, 1.5] for conf in [0.5, 1.0]
POSITION_SIZE_BASE: float = 0.5
INITIAL_PORTFOLIO_VALUE: float = 100_000.0

# ---------------------------------------------------------------------------
# Exit Logic (§9.3)
# ---------------------------------------------------------------------------
STOP_LOSS_PCT_DEFAULT: float = 0.10    # 10% stop-loss for standard positions
STOP_LOSS_PCT_MEME: float = 0.15       # 15% for meme stocks (GME, AMC, etc.)
TAKE_PROFIT_PARTIAL_PCT: float = 0.05  # partial exit at +5%
TAKE_PROFIT_FULL_PCT_MIN: float = 0.10
TAKE_PROFIT_FULL_PCT_MAX: float = 0.15
MAX_HOLDING_DAYS: int = 5              # max 5 trading days before forced exit
CONFIDENCE_EXIT_THRESHOLD: float = 0.50  # exit if daily confidence refresh drops below this

# ---------------------------------------------------------------------------
# Correlation Control (§9.4)
# ---------------------------------------------------------------------------
MAX_HOLDING_CORRELATION: float = 0.75  # reject trade if corr > 0.75 with existing holding
CORRELATION_LOOKBACK_DAYS: int = 20

# ---------------------------------------------------------------------------
# Market Regime (§9.5)
# ---------------------------------------------------------------------------
REGIME_SLOPE_WINDOW: int = 10          # days for linear regression slope
REGIME_BULL_REDUCE = 1.0               # full exposure
REGIME_BEAR_REDUCE = 0.5               # 50% exposure reduction
REGIME_CHOPPY_REDUCE = 0.30            # 70% frequency reduction (→ 30% of normal)

# ---------------------------------------------------------------------------
# IC Thresholds — Phase 0 Go/No-Go (§19)
# ---------------------------------------------------------------------------
IC_ABORT_THRESHOLD: float = 0.03      # median IC below this → ABORT, rethink thesis
IC_WEAK_SIGNAL: float = 0.05
IC_MEANINGFUL: float = 0.10
IC_STRONG: float = 0.15
IC_OVERFIT_WARNING: float = 0.25      # IC above this in backtest → assume leakage first

# ---------------------------------------------------------------------------
# Backtesting (§10.2)
# ---------------------------------------------------------------------------
BACKTEST_SLIPPAGE_MIN: float = 0.0005
BACKTEST_SLIPPAGE_MAX: float = 0.002
BACKTEST_FEE_PCT: float = 0.0005       # 0.05% per leg
BACKTEST_LIQUIDITY_PCT: float = 0.01   # max 1% of daily volume per trade
BACKTEST_RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Hard Stop Conditions (§15)
# ---------------------------------------------------------------------------
FINBERT_MAX_FAILURE_RATE: float = 0.05      # >5% failure rate → halt batch
MARKET_DATA_MAX_MISSING_RATE: float = 0.30  # >30% missing fields → reject ticker/date

# ---------------------------------------------------------------------------
# Phase 1 — Research Pipeline  (shorter aliases used by pipeline/ scripts)
# ---------------------------------------------------------------------------
# Data quality
MIN_POST_COUNT: int = MIN_POST_COUNT_1D        # 3
MIN_CONFIDENCE: float = 0.8                   # FinBERT confidence gate for avg_sentiment_hc
MAX_RETURN_WINSORISE: float = 0.50            # clip |target_return_5d| > 50%

# Model selection
MIN_IC_THRESHOLD: float = IC_ABORT_THRESHOLD  # 0.03
REDDIT_ADDS_VALUE_IC: float = 0.005           # min IC improvement for Reddit to "add value"
MIN_PRED_RETURN: float = 0.01                 # 1% min predicted return to trade

# Backtest mechanics
STARTING_CAPITAL: float = 1_000.0
MAX_POSITIONS: int = 3
HOLD_DAYS: int = 5
SLIPPAGE: float = 0.001                       # 0.1% per fill
FEE_PER_LEG: float = 0.0005                  # 0.05% per leg

# Statistical validation
N_PERMUTATIONS: int = 100
N_BOOTSTRAP: int = 50
PVALUE_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Phase 2 — Experiment A: Attention Filter
# ---------------------------------------------------------------------------
ATTENTION_FILTER_MIN_POSTS: int = 10
ATTENTION_FILTER_MIN_GROWTH: float = 0.3

# ---------------------------------------------------------------------------
# Phase 2 — Experiment B: Regime Detection
# ---------------------------------------------------------------------------
REGIME_LOOKBACK_DAYS: int = 60
REGIME_MIN_ROWS: int = 20
REGIME_POSITIVE_THRESHOLD: float = 0.05
REGIME_NEGATIVE_THRESHOLD: float = -0.05

# ---------------------------------------------------------------------------
# Phase 2 — Experiment C: Expanded Dataset
# ---------------------------------------------------------------------------
EXPANDED_PARQUET_PATH: str = "data/raw/merged_with_sentiment_expanded.parquet"
EXPANDED_FEATURES_PATH: str = "data/features/features_expanded.parquet"

# ---------------------------------------------------------------------------
# Phase 2 — Experiment Decision Criteria
# ---------------------------------------------------------------------------
EXPERIMENT_MIN_IC: float = 0.05          # must exceed to be a valid winner
EXPERIMENT_MIN_SHARPE: float = 1.0       # must exceed to be a valid winner

TAKE_PROFIT_CAP: float = 0.15            # close position early if unrealized gain >= 15%

# ---------------------------------------------------------------------------
# Phase 3 Feature Set — 11 features (8 market + 3 attention)
# L1 Granger test: sentiment 0/6 years significant → 6 sentiment features dropped.
# L3 family validation: pruning rejected → returns_20d/rsi_14/atr_14/mention_growth_1d
#   re-included from original 17. CLEAN_FEATURES alias kept for backward compat.
# ---------------------------------------------------------------------------
PHASE3_FEATURES: list = [
    # Market family (8)
    'returns_1d',
    'returns_5d',
    'returns_20d',
    'rsi_14',
    'atr_14',
    'relative_volume',
    'dist_from_20ma',
    'dist_from_50ma',
    # Attention family (3)
    'post_count_1d',
    'mention_growth_1d',
    'mention_growth_7d',
]

# Sentiment features dropped per L1 Granger (0/6 years significant)
SENTIMENT_FEATURES_DROPPED: list = [
    'avg_sentiment_1d',
    'avg_sentiment_3d',
    'weighted_sentiment',
    'sentiment_std',
    'sentiment_accel',
    'bullish_ratio',
]

CLEAN_FEATURES = PHASE3_FEATURES  # backward-compat alias

# Tickers with insufficient training rows (<30) or leakage risk
DROP_TICKERS: list = ['ASTS', 'LCID', 'MSTR', 'RIOT', 'RIVN', 'SMCI', 'WMT']
