"""
Module: features/reddit_features.py
Purpose: Compute per-ticker Reddit sentiment features from aggregated post data.
         All windows computed relative to T (market open on T, not close).
         See §5.1 for full feature specification.
Phase: 2 — Feature Store + Alignment Validation
Dependencies: config/thresholds.py, features/alignment.py, utils/logger.py
Last modified: 2026-06-10
"""

import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config.thresholds import MIN_POST_COUNT_1D, MIN_AVG_SENTIMENT_CONFIDENCE
from features.alignment import get_reddit_window
from utils.logger import get_logger

log = get_logger(__name__)


def compute_reddit_features(
    reddit_df: pd.DataFrame,
    ticker: str,
    row_date: date,
) -> Optional[dict]:
    """
    Compute all §5.1 Reddit features for a (ticker, date) row.

    Windows are strictly before market open on row_date (enforced by alignment.py).
    Returns None if the row fails the quality filter (§5.4).

    The reddit_df must have columns:
        ticker, timestamp (UTC), sentiment_score (float or NaN), sentiment_confidence,
        upvotes, comment_count, author (author ID), post_id

    Args:
        reddit_df: Full Reddit posts with FinBERT scores already applied
        ticker: Ticker to compute features for
        row_date: Feature row date T

    Returns:
        Dict of feature values, or None if quality filter rejects this row.
    """
    # --- Fetch windowed data (alignment enforced inside) ---
    posts_1d = get_reddit_window(reddit_df, row_date, ticker, window_hours=24)
    posts_3d = get_reddit_window(reddit_df, row_date, ticker, window_hours=72)
    posts_5d = get_reddit_window(reddit_df, row_date, ticker, window_hours=120)

    # --- Quality filter: minimum post count (§5.4) ---
    if len(posts_1d) < MIN_POST_COUNT_1D:
        log.info(
            "reddit_feature_excluded",
            ticker=ticker,
            date=str(row_date),
            reason="post_count_1d_below_minimum",
            post_count_1d=len(posts_1d),
            threshold=MIN_POST_COUNT_1D,
        )
        return None

    # --- Post counts ---
    post_count_1d = float(len(posts_1d))
    post_count_3d = float(len(posts_3d))
    post_count_5d = float(len(posts_5d))

    # --- Sentiment features ---
    # Only include posts where FinBERT returned a valid score (§3.3: never fill NaN)
    valid_1d = posts_1d.dropna(subset=["sentiment_score"])
    valid_3d = posts_3d.dropna(subset=["sentiment_score"])

    avg_sentiment_1d: Optional[float] = (
        float(valid_1d["sentiment_score"].mean()) if not valid_1d.empty else None
    )
    avg_sentiment_3d: Optional[float] = (
        float(valid_3d["sentiment_score"].mean()) if not valid_3d.empty else None
    )

    # Quality filter: average confidence check (§5.4)
    if "sentiment_confidence" in valid_1d.columns and not valid_1d.empty:
        avg_conf = float(valid_1d["sentiment_confidence"].mean())
        if avg_conf < MIN_AVG_SENTIMENT_CONFIDENCE:
            log.info(
                "reddit_feature_excluded",
                ticker=ticker,
                date=str(row_date),
                reason="avg_sentiment_confidence_below_minimum",
                avg_confidence=round(avg_conf, 4),
                threshold=MIN_AVG_SENTIMENT_CONFIDENCE,
            )
            return None

    # Weighted sentiment: sum(s * log(upvotes+1)) / n
    weighted_sentiment_1d: Optional[float] = None
    if not valid_1d.empty:
        weights = valid_1d["upvotes"].apply(lambda u: math.log(max(float(u), 0) + 1))
        total_weight = weights.sum()
        if total_weight > 0:
            weighted_sentiment_1d = float(
                (valid_1d["sentiment_score"] * weights).sum() / len(valid_1d)
            )

    # Sentiment acceleration: avg_1d - avg_3d (None if either is None)
    sentiment_acceleration: Optional[float] = None
    if avg_sentiment_1d is not None and avg_sentiment_3d is not None:
        sentiment_acceleration = avg_sentiment_1d - avg_sentiment_3d

    # Mention growth: post_count_1d / (post_count_3d + 1)
    mention_growth = post_count_1d / (post_count_3d + 1.0)

    # Upvote velocity: sum(upvotes) / 24 hours
    upvote_velocity_1d = float(posts_1d["upvotes"].sum()) / 24.0

    # Unique authors
    unique_authors_1d = float(posts_1d["author"].nunique()) if "author" in posts_1d.columns else None

    # Engagement score: sum(upvotes + comments) / post_count
    engagement_score: Optional[float] = None
    if post_count_1d > 0 and "comment_count" in posts_1d.columns:
        total_engagement = (posts_1d["upvotes"] + posts_1d["comment_count"]).sum()
        engagement_score = float(total_engagement) / post_count_1d

    features = {
        "ticker": ticker,
        "date": str(row_date),
        "post_count_1d": post_count_1d,
        "post_count_3d": post_count_3d,
        "post_count_5d": post_count_5d,
        "avg_sentiment_1d": avg_sentiment_1d,
        "avg_sentiment_3d": avg_sentiment_3d,
        "weighted_sentiment_1d": weighted_sentiment_1d,
        "sentiment_acceleration": sentiment_acceleration,
        "mention_growth": mention_growth,
        "upvote_velocity_1d": upvote_velocity_1d,
        "unique_authors_1d": unique_authors_1d,
        "engagement_score": engagement_score,
    }

    log.info(
        "reddit_features_computed",
        ticker=ticker,
        date=str(row_date),
        post_count_1d=int(post_count_1d),
        avg_sentiment_1d=round(avg_sentiment_1d, 4) if avg_sentiment_1d is not None else None,
        sentiment_acceleration=round(sentiment_acceleration, 4) if sentiment_acceleration is not None else None,
    )
    return features
