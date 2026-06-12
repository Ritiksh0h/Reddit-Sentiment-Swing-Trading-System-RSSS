"""
Module: features/alignment.py
Purpose: Time-align Reddit posts and market data to produce a leak-free feature matrix.
         THE MOST CRITICAL FILE IN THE SYSTEM.

         Any bug here invalidates all downstream work. Read §3.1 before touching this file.

Phase: 2 — Feature Store + Alignment Validation
Dependencies: config/settings.py, config/thresholds.py, utils/logger.py, utils/time_utils.py
Last modified: 2026-06-10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INVARIANT — ENFORCED BY ALL FUNCTIONS IN THIS MODULE:

  For every feature row (ticker, date=T):
    • Reddit posts:  timestamp <  market_open(T)  [strictly before T opens]
    • Market data:   date      <  T               [strictly prior trading days]
    • Return labels: use close[T+1], close[T+3], close[T+5]  — never close[T] or earlier

  Violation = data leakage = system must halt with DataIntegrityError.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before modifying this file:
  1. Run: pytest tests/test_alignment.py -v
  2. Make your change
  3. Run: pytest tests/test_alignment.py -v
  4. Show before/after output
"""

import json
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from config.settings import MARKET_TIMEZONE
from utils.logger import get_logger
from utils.time_utils import market_open_utc, to_utc

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class DataIntegrityError(Exception):
    """
    Raised when any time alignment check detects future data leaking into features.
    Pipeline must halt — never catch and continue.
    """


# ---------------------------------------------------------------------------
# Feature Window Cutoff
# ---------------------------------------------------------------------------


def get_feature_cutoff(row_date: date) -> datetime:
    """
    Return the hard upper cutoff for all feature data used on row_date.

    The cutoff is 09:30 ET (market open) on row_date. Posts that arrive
    AFTER the previous day's close but BEFORE market open on T are excluded
    — this is the conservative safe choice.

    Using market close (16:00) instead would allow same-day Reddit chatter
    that reacts to intraday price moves to leak forward-looking information.

    Args:
        row_date: The trading date this feature row represents

    Returns:
        UTC-aware datetime. All Reddit timestamps must be < this value.
    """
    return market_open_utc(row_date)


# ---------------------------------------------------------------------------
# Validation (raises on violation — never returns False)
# ---------------------------------------------------------------------------


def validate_reddit_window(
    reddit_window: pd.DataFrame,
    row_date: date,
    ticker: str,
) -> None:
    """
    Assert that no Reddit post in the window has timestamp >= feature cutoff.

    Raises DataIntegrityError immediately on any violation. Never continues.

    Args:
        reddit_window: DataFrame of Reddit posts with 'timestamp' column (UTC str or datetime)
        row_date: Feature row date
        ticker: Ticker symbol for error context

    Raises:
        DataIntegrityError: If any post timestamp falls on or after the cutoff.
    """
    if reddit_window.empty:
        return

    cutoff = get_feature_cutoff(row_date)
    timestamps = pd.to_datetime(reddit_window["timestamp"], utc=True)
    future_mask = timestamps >= pd.Timestamp(cutoff)

    if future_mask.any():
        bad = reddit_window[future_mask.values]
        first_bad = bad["timestamp"].iloc[0]
        detail = (
            f"post timestamp {first_bad} found in feature window for date {row_date} "
            f"(cutoff={cutoff.isoformat()})"
        )
        log.error(
            "data_integrity_violation",
            reason="future_sentiment_detected",
            ticker=ticker,
            date=str(row_date),
            num_violations=int(future_mask.sum()),
            first_violation=str(first_bad),
            cutoff=cutoff.isoformat(),
        )
        raise DataIntegrityError(
            json.dumps({
                "status": "FAILED",
                "reason": "data_integrity_violation",
                "detail": detail,
                "ticker": ticker,
                "date": str(row_date),
            })
        )


def validate_market_window(
    market_window: pd.DataFrame,
    row_date: date,
    ticker: str,
) -> None:
    """
    Assert that no market row in the window has date >= row_date.

    Args:
        market_window: OHLCV DataFrame indexed by date
        row_date: Feature row date
        ticker: Ticker symbol for error context

    Raises:
        DataIntegrityError: If any market date >= row_date.
    """
    if market_window.empty:
        return

    dates = pd.to_datetime(market_window.index).normalize()
    future_mask = dates >= pd.Timestamp(row_date)

    if future_mask.any():
        first_bad = dates[future_mask][0].date()
        detail = (
            f"market data date {first_bad} found in feature window for date {row_date}"
        )
        log.error(
            "data_integrity_violation",
            reason="future_market_data_detected",
            ticker=ticker,
            date=str(row_date),
            first_violation=str(first_bad),
        )
        raise DataIntegrityError(
            json.dumps({
                "status": "FAILED",
                "reason": "data_integrity_violation",
                "detail": detail,
                "ticker": ticker,
                "date": str(row_date),
            })
        )


# ---------------------------------------------------------------------------
# Windowed Data Access
# ---------------------------------------------------------------------------


def get_reddit_window(
    reddit_df: pd.DataFrame,
    row_date: date,
    ticker: str,
    window_hours: int,
) -> pd.DataFrame:
    """
    Return Reddit posts for a ticker in the window [T - window_hours, cutoff(T)).

    CRITICAL: Upper bound is STRICT less-than. Never <=.

    Args:
        reddit_df: Full Reddit posts DataFrame. Must have columns:
                   'ticker', 'timestamp' (UTC-parseable string or datetime)
        row_date: The feature row date T
        ticker: Ticker to filter on
        window_hours: Hours to look back from the feature cutoff

    Returns:
        Filtered DataFrame. May be empty — caller must handle that.
        Never returns None; raises DataIntegrityError on leakage.

    Raises:
        DataIntegrityError: If any post in the window violates the time cutoff.
    """
    cutoff = get_feature_cutoff(row_date)
    window_start = pd.Timestamp(cutoff) - pd.Timedelta(hours=window_hours)

    timestamps = pd.to_datetime(reddit_df["timestamp"], utc=True)
    mask = (
        (reddit_df["ticker"] == ticker)
        & (timestamps >= window_start)
        & (timestamps < pd.Timestamp(cutoff))  # STRICT less-than
    )

    window = reddit_df[mask].copy()
    validate_reddit_window(window, row_date, ticker)
    return window


def get_market_window(
    market_df: pd.DataFrame,
    row_date: date,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Return the last lookback_days market rows strictly before row_date.

    CRITICAL: Upper bound is STRICT less-than. Never <=.

    Args:
        market_df: OHLCV DataFrame indexed by date
        row_date: Feature row date T
        lookback_days: Number of prior trading days to include

    Returns:
        Filtered DataFrame. May be empty. Raises on leakage.

    Raises:
        DataIntegrityError: If any market row is on or after row_date.
    """
    dates = pd.to_datetime(market_df.index).normalize()
    row_ts = pd.Timestamp(row_date)

    prior_mask = dates < row_ts  # STRICT less-than
    prior_dates = dates[prior_mask]

    if len(prior_dates) == 0:
        return market_df.iloc[0:0].copy()

    # DatetimeIndex supports plain integer indexing, not .iloc
    start_ts = (
        prior_dates[-lookback_days]
        if len(prior_dates) >= lookback_days
        else prior_dates[0]
    )

    window_mask = (dates >= start_ts) & (dates < row_ts)
    # dates is a DatetimeIndex; comparison already returns a numpy bool array
    window = market_df[window_mask].copy()

    validate_market_window(window, row_date, "N/A")
    return window


# ---------------------------------------------------------------------------
# Return Label Computation
# ---------------------------------------------------------------------------


def compute_return_labels(
    market_df: pd.DataFrame,
    row_date: date,
    ticker: str,
) -> dict[str, Optional[float]]:
    """
    Compute forward return labels for row_date using strictly future prices.

    Labels:
        return_1d = (close[T+1] - close[T]) / close[T]
        return_3d = (close[T+3] - close[T]) / close[T]
        return_5d = (close[T+5] - close[T]) / close[T]

    T+N refers to the Nth trading day after T in the market_df index.

    Args:
        market_df: Full OHLCV DataFrame indexed by date (must include T and T+N rows)
        row_date: The feature row date T
        ticker: For logging only

    Returns:
        Dict with keys: return_1d, return_3d, return_5d.
        Value is None if insufficient future data exists (row excluded downstream).
    """
    dates = pd.to_datetime(market_df.index).normalize()
    row_ts = pd.Timestamp(row_date)

    # dates == row_ts returns a numpy bool array when index is DatetimeIndex; no .values needed
    date_positions = (dates == row_ts).nonzero()[0]
    if len(date_positions) == 0:
        log.warning("label_base_date_missing", ticker=ticker, date=str(row_date))
        return {"return_1d": None, "return_3d": None, "return_5d": None}

    row_idx = int(date_positions[0])
    base_close = float(market_df.iloc[row_idx]["close"])

    labels: dict[str, Optional[float]] = {}
    for label_name, offset in [("return_1d", 1), ("return_3d", 3), ("return_5d", 5)]:
        future_idx = row_idx + offset
        if future_idx >= len(market_df):
            log.warning(
                "label_insufficient_future_data",
                ticker=ticker,
                date=str(row_date),
                horizon=label_name,
                available=len(market_df) - row_idx - 1,
                needed=offset,
            )
            labels[label_name] = None
        else:
            future_close = float(market_df.iloc[future_idx]["close"])
            labels[label_name] = (future_close - base_close) / base_close

    return labels
