"""
Module: tests/test_alignment.py
Purpose: Mandatory leakage detection tests. These must pass before ANY model training.
         Tests run on synthetic data with known leakage injected to confirm detection.
Phase: 2 — Feature Store + Alignment Validation
Dependencies: features/alignment.py, utils/time_utils.py
Last modified: 2026-06-10

Run with:
    pytest tests/test_alignment.py -v

MUST PASS before touching features/alignment.py.
"""

import json
from datetime import date, datetime, timezone, timedelta

import pandas as pd
import pytest

from features.alignment import (
    DataIntegrityError,
    compute_return_labels,
    get_feature_cutoff,
    get_market_window,
    get_reddit_window,
    validate_market_window,
    validate_reddit_window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reddit_df(timestamps_utc: list[datetime], ticker: str = "NVDA") -> pd.DataFrame:
    """Build a minimal Reddit DataFrame with the given UTC timestamps."""
    return pd.DataFrame({
        "post_id": [f"p{i}" for i in range(len(timestamps_utc))],
        "ticker": ticker,
        "timestamp": [ts.isoformat() for ts in timestamps_utc],
        "upvotes": [100] * len(timestamps_utc),
        "comment_count": [10] * len(timestamps_utc),
        "author": [f"user{i}" for i in range(len(timestamps_utc))],
        "sentiment_score": [0.5] * len(timestamps_utc),
        "sentiment_confidence": [0.8] * len(timestamps_utc),
    })


def make_market_df(dates: list[date], base_price: float = 100.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with the given dates."""
    prices = [base_price + i for i in range(len(dates))]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * len(dates),
        },
        index=pd.to_datetime(dates),
    )


# ---------------------------------------------------------------------------
# get_feature_cutoff
# ---------------------------------------------------------------------------

class TestGetFeatureCutoff:
    def test_returns_utc_aware_datetime(self):
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        assert cutoff.tzinfo is not None

    def test_cutoff_is_before_market_close(self):
        """Cutoff (09:30 ET) must be strictly before 16:00 ET close."""
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        # 09:30 ET = 13:30 UTC (EST) or 14:30 UTC (EDT)
        # 16:00 ET = 21:00 UTC (EST) or 20:00 UTC (EDT)
        assert cutoff.hour < 21  # definitely before market close UTC equivalent

    def test_cutoff_date_matches_row_date(self):
        """Cutoff should fall on the same calendar date as row_date (in ET)."""
        import pytz
        row_date = date(2024, 6, 10)
        cutoff = get_feature_cutoff(row_date)
        et_cutoff = cutoff.astimezone(pytz.timezone("America/New_York"))
        assert et_cutoff.date() == row_date


# ---------------------------------------------------------------------------
# validate_reddit_window — clean data should not raise
# ---------------------------------------------------------------------------

class TestValidateRedditWindowClean:
    def test_empty_window_does_not_raise(self):
        """Empty window is valid — caller handles missing data."""
        df = make_reddit_df([])
        validate_reddit_window(df, date(2024, 3, 15), "NVDA")  # should not raise

    def test_prior_day_posts_do_not_raise(self):
        """Posts from the previous day must not trigger leakage detection."""
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        safe_ts = cutoff - timedelta(hours=2)  # 2 hours before cutoff — safe
        df = make_reddit_df([safe_ts])
        validate_reddit_window(df, date(2024, 3, 15), "NVDA")  # should not raise

    def test_multiple_safe_posts_do_not_raise(self):
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        safe_timestamps = [cutoff - timedelta(hours=h) for h in range(1, 25)]
        df = make_reddit_df(safe_timestamps)
        validate_reddit_window(df, date(2024, 3, 15), "NVDA")  # should not raise


# ---------------------------------------------------------------------------
# validate_reddit_window — leakage should raise DataIntegrityError
# ---------------------------------------------------------------------------

class TestValidateRedditWindowLeakage:
    def test_same_cutoff_timestamp_raises(self):
        """A post exactly at cutoff (<=) must be rejected. We use strict <."""
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        df = make_reddit_df([cutoff])  # exactly at cutoff — leakage
        with pytest.raises(DataIntegrityError) as exc_info:
            validate_reddit_window(df, date(2024, 3, 15), "NVDA")
        error_data = json.loads(str(exc_info.value))
        assert error_data["status"] == "FAILED"
        assert error_data["reason"] == "data_integrity_violation"
        assert error_data["ticker"] == "NVDA"

    def test_post_after_cutoff_raises(self):
        """A post after market open on T must be rejected."""
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        future_ts = cutoff + timedelta(hours=1)
        df = make_reddit_df([future_ts])
        with pytest.raises(DataIntegrityError):
            validate_reddit_window(df, date(2024, 3, 15), "NVDA")

    def test_mixed_safe_and_leaky_raises(self):
        """Even one leaky post in the window must trigger the error."""
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        safe_ts = cutoff - timedelta(hours=5)
        leaky_ts = cutoff + timedelta(minutes=30)
        df = make_reddit_df([safe_ts, leaky_ts])
        with pytest.raises(DataIntegrityError):
            validate_reddit_window(df, date(2024, 3, 15), "NVDA")

    def test_next_day_post_raises(self):
        """A post from the next calendar day obviously leaks."""
        next_day_ts = datetime(2024, 3, 16, 10, 0, 0, tzinfo=timezone.utc)
        df = make_reddit_df([next_day_ts])
        with pytest.raises(DataIntegrityError):
            validate_reddit_window(df, date(2024, 3, 15), "NVDA")

    def test_error_contains_structured_json(self):
        """DataIntegrityError message must be parseable JSON with required fields."""
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        df = make_reddit_df([cutoff + timedelta(hours=1)])
        with pytest.raises(DataIntegrityError) as exc_info:
            validate_reddit_window(df, date(2024, 3, 15), "AAPL")
        error_data = json.loads(str(exc_info.value))
        assert "status" in error_data
        assert "reason" in error_data
        assert "detail" in error_data
        assert "ticker" in error_data
        assert "date" in error_data


# ---------------------------------------------------------------------------
# validate_market_window
# ---------------------------------------------------------------------------

class TestValidateMarketWindow:
    def test_prior_dates_do_not_raise(self):
        dates = [date(2024, 3, 11), date(2024, 3, 12), date(2024, 3, 13), date(2024, 3, 14)]
        df = make_market_df(dates)
        validate_market_window(df, date(2024, 3, 15), "NVDA")  # should not raise

    def test_same_day_raises(self):
        """Market data for row_date itself must not be in the feature window."""
        dates = [date(2024, 3, 14), date(2024, 3, 15)]  # includes row_date
        df = make_market_df(dates)
        with pytest.raises(DataIntegrityError):
            validate_market_window(df, date(2024, 3, 15), "NVDA")

    def test_future_date_raises(self):
        dates = [date(2024, 3, 14), date(2024, 3, 16)]
        df = make_market_df(dates)
        with pytest.raises(DataIntegrityError):
            validate_market_window(df, date(2024, 3, 15), "NVDA")

    def test_empty_window_does_not_raise(self):
        df = make_market_df([])
        validate_market_window(df, date(2024, 3, 15), "NVDA")  # should not raise


# ---------------------------------------------------------------------------
# get_reddit_window
# ---------------------------------------------------------------------------

class TestGetRedditWindow:
    def test_filters_by_ticker(self):
        """Posts for other tickers must not appear in the window."""
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        nvda_ts = cutoff - timedelta(hours=5)
        tsla_ts = cutoff - timedelta(hours=3)
        df = pd.concat([
            make_reddit_df([nvda_ts], ticker="NVDA"),
            make_reddit_df([tsla_ts], ticker="TSLA"),
        ], ignore_index=True)
        window = get_reddit_window(df, date(2024, 3, 15), "NVDA", window_hours=24)
        assert all(window["ticker"] == "NVDA")
        assert len(window) == 1

    def test_excludes_posts_outside_window(self):
        """Posts older than window_hours must be excluded."""
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        recent = cutoff - timedelta(hours=5)
        old = cutoff - timedelta(hours=30)  # outside 24h window
        df = make_reddit_df([recent, old])
        window = get_reddit_window(df, date(2024, 3, 15), "NVDA", window_hours=24)
        assert len(window) == 1

    def test_excludes_post_exactly_at_cutoff(self):
        """
        A post timestamped exactly at the cutoff must be excluded from the window
        (strict < upper bound), NOT included. The window should be empty.

        Note: get_reddit_window uses strict < to FILTER — it does not raise if the
        upstream DataFrame contains posts at/after the cutoff; it simply excludes them.
        validate_reddit_window (tested above) is the function that raises on leakage
        when called on a pre-constructed window.
        """
        cutoff = get_feature_cutoff(date(2024, 3, 15))
        df = make_reddit_df([cutoff])  # post exactly at cutoff
        window = get_reddit_window(df, date(2024, 3, 15), "NVDA", window_hours=24)
        # The post is AT the cutoff, so strictly-less-than filtering excludes it
        assert len(window) == 0, (
            "A post at the feature cutoff must be excluded from the window (strict <)."
        )


# ---------------------------------------------------------------------------
# get_market_window
# ---------------------------------------------------------------------------

class TestGetMarketWindow:
    def test_excludes_row_date(self):
        """Row_date itself must never appear in the market window."""
        dates = [
            date(2024, 3, 11), date(2024, 3, 12), date(2024, 3, 13),
            date(2024, 3, 14), date(2024, 3, 15),  # row_date included in full df
        ]
        df = make_market_df(dates)
        window = get_market_window(df, date(2024, 3, 15), lookback_days=10)
        window_dates = pd.to_datetime(window.index).normalize()
        assert pd.Timestamp(date(2024, 3, 15)) not in window_dates

    def test_respects_lookback_limit(self):
        dates = [date(2024, 3, 1) + timedelta(days=i) for i in range(20)]
        df = make_market_df(dates)
        row_date = date(2024, 3, 18)
        window = get_market_window(df, row_date, lookback_days=5)
        assert len(window) <= 5

    def test_empty_when_no_prior_data(self):
        dates = [date(2024, 3, 15)]  # only row_date — no prior data
        df = make_market_df(dates)
        window = get_market_window(df, date(2024, 3, 15), lookback_days=5)
        assert len(window) == 0


# ---------------------------------------------------------------------------
# compute_return_labels
# ---------------------------------------------------------------------------

class TestComputeReturnLabels:
    def test_labels_use_future_data(self):
        """return_1d must use close[T+1], not close[T] or prior."""
        dates = [date(2024, 3, 1) + timedelta(days=i) for i in range(10)]
        # Prices: 100, 101, 102, ..., 109
        df = make_market_df(dates, base_price=100.0)
        row_date = date(2024, 3, 4)  # index 3, close=103

        labels = compute_return_labels(df, row_date, "NVDA")

        assert labels["return_1d"] is not None
        # close[T] = 103, close[T+1] = 104
        expected_1d = (104.0 - 103.0) / 103.0
        assert abs(labels["return_1d"] - expected_1d) < 1e-9

    def test_returns_none_when_insufficient_future_data(self):
        dates = [date(2024, 3, 1) + timedelta(days=i) for i in range(5)]
        df = make_market_df(dates, base_price=100.0)
        row_date = date(2024, 3, 4)  # only 1 future row — return_3d and return_5d must be None
        labels = compute_return_labels(df, row_date, "NVDA")
        assert labels["return_3d"] is None
        assert labels["return_5d"] is None

    def test_returns_none_when_row_date_missing(self):
        dates = [date(2024, 3, 1), date(2024, 3, 2)]
        df = make_market_df(dates)
        labels = compute_return_labels(df, date(2024, 3, 5), "NVDA")
        assert labels["return_1d"] is None
        assert labels["return_3d"] is None
        assert labels["return_5d"] is None

    def test_all_labels_computed_with_sufficient_data(self):
        dates = [date(2024, 3, 1) + timedelta(days=i) for i in range(10)]
        df = make_market_df(dates)
        row_date = date(2024, 3, 3)  # index 2; need +5 = index 7 which exists
        labels = compute_return_labels(df, row_date, "NVDA")
        assert labels["return_1d"] is not None
        assert labels["return_3d"] is not None
        assert labels["return_5d"] is not None

    def test_injected_leakage_not_in_labels(self):
        """
        Synthetic leakage test: confirm labels use T+N, not T-N.
        If labels were backwards (using prior days), they'd be negative
        for an ascending price series when they should be positive.

        Date list: March 1–10 (indices 0–9), strictly ascending prices 100–109.
        row_date = March 5 (index 4, price=104).
          T+1 = index 5 = 105  →  return_1d = (105-104)/104 > 0
          T+3 = index 7 = 107  →  return_3d = (107-104)/104 > 0
          T+5 = index 9 = 109  →  return_5d = (109-104)/104 > 0  (index 9 EXISTS in 10-row df)
        """
        dates = [date(2024, 3, 1) + timedelta(days=i) for i in range(10)]
        # Strictly ascending prices: 100, 101, ..., 109
        df = make_market_df(dates, base_price=100.0)
        row_date = date(2024, 3, 5)  # index 4, price = 104

        labels = compute_return_labels(df, row_date, "NVDA")

        # All forward returns must be positive (prices strictly increase forward)
        assert labels["return_1d"] is not None and labels["return_1d"] > 0, \
            "return_1d should be positive for ascending prices (uses T+1, not T-1)"
        assert labels["return_3d"] is not None and labels["return_3d"] > 0, \
            "return_3d should be positive for ascending prices (uses T+3, not T-3)"
        assert labels["return_5d"] is not None and labels["return_5d"] > 0, \
            "return_5d should be positive for ascending prices (index 9 exists in 10-row df)"
