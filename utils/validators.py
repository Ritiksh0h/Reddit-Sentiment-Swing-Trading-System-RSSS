"""
Module: utils/validators.py
Purpose: Input/output schema enforcement via pandera. Rejects invalid rows rather
         than silently corrupting the dataset (§4.3).
Phase: 2 — Feature Store + Alignment Validation
Dependencies: pandera, pandas
Last modified: 2026-06-10
"""

from typing import Optional

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------


class RedditPostSchema(pa.DataFrameModel):
    """Schema for raw Reddit posts after ingestion."""

    post_id: Series[str] = pa.Field(nullable=False)
    subreddit: Series[str] = pa.Field(nullable=False)
    title: Series[str] = pa.Field(nullable=False)
    body: Series[Optional[str]] = pa.Field(nullable=True)
    upvotes: Series[int] = pa.Field(ge=0, nullable=False)
    comment_count: Series[int] = pa.Field(ge=0, nullable=False)
    author_karma: Series[Optional[int]] = pa.Field(nullable=True)
    created_utc: Series[float] = pa.Field(nullable=False)  # Unix timestamp

    class Config:
        strict = False  # allow extra columns


class TickerMentionSchema(pa.DataFrameModel):
    """Schema for ticker mentions extracted from posts."""

    post_id: Series[str] = pa.Field(nullable=False)
    ticker: Series[str] = pa.Field(str_matches=r"^[A-Z]{1,5}$", nullable=False)
    timestamp: Series[str] = pa.Field(nullable=False)  # ISO UTC string
    upvotes: Series[int] = pa.Field(ge=0)
    sentiment_score: Series[Optional[float]] = pa.Field(
        nullable=True, ge=-1.0, le=1.0
    )
    sentiment_label: Series[Optional[str]] = pa.Field(nullable=True)
    sentiment_confidence: Series[Optional[float]] = pa.Field(
        nullable=True, ge=0.0, le=1.0
    )
    model_used: Series[Optional[str]] = pa.Field(nullable=True)

    class Config:
        strict = False


class MarketOHLCVSchema(pa.DataFrameModel):
    """Schema for daily OHLCV market data."""

    open: Series[float] = pa.Field(gt=0, nullable=False)
    high: Series[float] = pa.Field(gt=0, nullable=False)
    low: Series[float] = pa.Field(gt=0, nullable=False)
    close: Series[float] = pa.Field(gt=0, nullable=False)
    volume: Series[float] = pa.Field(ge=0, nullable=False)

    class Config:
        strict = False


class FeatureRowSchema(pa.DataFrameModel):
    """Schema for a fully assembled feature row before model training."""

    ticker: Series[str] = pa.Field(nullable=False)
    date: Series[str] = pa.Field(nullable=False)

    # Reddit features
    post_count_1d: Series[float] = pa.Field(ge=0, nullable=True)
    post_count_3d: Series[float] = pa.Field(ge=0, nullable=True)
    post_count_5d: Series[float] = pa.Field(ge=0, nullable=True)
    avg_sentiment_1d: Series[Optional[float]] = pa.Field(
        ge=-1.0, le=1.0, nullable=True
    )
    avg_sentiment_3d: Series[Optional[float]] = pa.Field(
        ge=-1.0, le=1.0, nullable=True
    )
    weighted_sentiment_1d: Series[Optional[float]] = pa.Field(nullable=True)
    sentiment_acceleration: Series[Optional[float]] = pa.Field(nullable=True)
    mention_growth: Series[Optional[float]] = pa.Field(nullable=True)
    upvote_velocity_1d: Series[Optional[float]] = pa.Field(ge=0, nullable=True)
    unique_authors_1d: Series[Optional[float]] = pa.Field(ge=0, nullable=True)
    engagement_score: Series[Optional[float]] = pa.Field(ge=0, nullable=True)

    # Market features
    rsi_14: Series[Optional[float]] = pa.Field(ge=0.0, le=100.0, nullable=True)
    relative_volume: Series[Optional[float]] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = False


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------


def validate_reddit_posts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Validate a DataFrame of Reddit posts against RedditPostSchema.

    Returns the valid subset and the number of rows dropped.
    Logs each dropped row — never drops silently.

    Args:
        df: Raw Reddit posts DataFrame

    Returns:
        Tuple of (valid_df, dropped_count)
    """
    try:
        validated = RedditPostSchema.validate(df, lazy=True)
        return validated, 0
    except pa.errors.SchemaErrors as e:
        failure_cases = e.failure_cases
        dropped_ids = failure_cases["index"].unique()
        n_dropped = len(dropped_ids)
        for _, row in failure_cases.iterrows():
            log.warning(
                "reddit_post_schema_violation",
                index=row.get("index"),
                column=row.get("column"),
                check=row.get("check"),
                failure_case=str(row.get("failure_case")),
            )
        valid_df = df.drop(index=dropped_ids, errors="ignore")
        log.info(
            "reddit_posts_validated",
            total=len(df),
            valid=len(valid_df),
            dropped=n_dropped,
        )
        return valid_df, n_dropped


def validate_market_ohlcv(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, int]:
    """
    Validate OHLCV data for a single ticker.

    Args:
        df: OHLCV DataFrame
        ticker: Ticker symbol (for logging)

    Returns:
        Tuple of (valid_df, dropped_count)
    """
    try:
        validated = MarketOHLCVSchema.validate(df, lazy=True)
        return validated, 0
    except pa.errors.SchemaErrors as e:
        failure_cases = e.failure_cases
        dropped_idx = failure_cases["index"].unique()
        n_dropped = len(dropped_idx)
        for _, row in failure_cases.iterrows():
            log.warning(
                "market_ohlcv_schema_violation",
                ticker=ticker,
                index=str(row.get("index")),
                column=row.get("column"),
                check=row.get("check"),
            )
        valid_df = df.drop(index=dropped_idx, errors="ignore")
        return valid_df, n_dropped
