"""
Module: pipeline/feature_schema.py
Purpose: Single source of truth for feature column names as they exist in
         data/features/features.parquet (output of 01_feature_builder.py).
         Imported by scripts 03–06. Change here if feature builder changes.

Last modified: 2026-06-11
"""

# ---------------------------------------------------------------------------
# Market features (from yfinance + pandas-ta)
# ---------------------------------------------------------------------------
MARKET_FEATURES: list[str] = [
    "rsi_14",           # RSI-14
    "atr_14",           # ATR-14
    "dist_from_20ma",   # % distance from 20-day SMA
    "dist_from_50ma",   # % distance from 50-day SMA
    "relative_volume",  # today vol / 20-day avg vol
    "returns_1d",       # 1-trading-day lagged return
    "returns_5d",       # 5-trading-day lagged return
    "returns_20d",      # 20-trading-day lagged return
    "volume",           # raw daily volume
    "close",            # closing price (for position sizing)
]

# ---------------------------------------------------------------------------
# Reddit features (from 01_feature_builder.py window aggregations)
# ---------------------------------------------------------------------------
REDDIT_FEATURES: list[str] = [
    "post_count_1d",        # posts in prior 24h
    "post_count_3d",        # posts in prior 72h
    "post_count_7d",        # posts in prior 168h
    "avg_sentiment_1d",     # avg FinBERT score, 24h window
    "avg_sentiment_3d",     # avg FinBERT score, 72h window
    "avg_sentiment_hc",     # high-confidence avg (conf >= 0.8)
    "mention_growth_1d",    # 24h / 7d mention growth rate
    "mention_growth_7d",    # 7d mention growth rate
    "total_upvotes_1d",     # total post upvotes, 24h
    "total_comments_1d",    # total comments, 24h
    "unique_authors_1d",    # unique Reddit authors, 24h
    "weighted_sentiment",   # upvote-weighted sentiment
    "bullish_ratio",        # fraction of bullish posts
    "sentiment_accel",      # sentiment acceleration (3d - 7d trend)
    "sentiment_std",        # sentiment standard deviation, 24h
]

# ---------------------------------------------------------------------------
# Target columns
# ---------------------------------------------------------------------------
TARGET_COL: str = "target_return_5d"
TARGET_COL_10D: str = "target_return_10d"  # for Phase 2B 10-day hold extension

# ---------------------------------------------------------------------------
# Ablation subsets (6 required by spec)
# ---------------------------------------------------------------------------
ABLATION_SUBSETS: dict[str, list[str]] = {
    "market_only": MARKET_FEATURES,
    "reddit_volume": [
        "post_count_1d", "post_count_3d", "post_count_7d",
        "mention_growth_1d", "mention_growth_7d",
        "total_upvotes_1d", "total_comments_1d", "unique_authors_1d",
    ],
    "reddit_sentiment": [
        "avg_sentiment_1d", "avg_sentiment_3d", "avg_sentiment_hc",
        "weighted_sentiment", "bullish_ratio",
        "sentiment_accel", "sentiment_std",
    ],
    "reddit_velocity": [
        "post_count_1d", "mention_growth_1d", "sentiment_accel",
        "total_upvotes_1d", "unique_authors_1d",
    ],
    "combined_core": MARKET_FEATURES + [
        "post_count_1d", "avg_sentiment_1d", "mention_growth_1d",
        "weighted_sentiment", "sentiment_accel",
    ],
    "combined_all": MARKET_FEATURES + REDDIT_FEATURES,
}
