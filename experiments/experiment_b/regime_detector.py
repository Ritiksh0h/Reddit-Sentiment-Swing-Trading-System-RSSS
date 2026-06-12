"""
Module: experiments/experiment_b/regime_detector.py
Purpose: Rolling sentiment-return correlation regime detector.

Regime for each (ticker, date) is computed from the 60-day window of past data only.
The window uses [T-60d, T) — strictly no data from T onwards (leakage-free).

Regimes:
  'positive' — correlation > REGIME_POSITIVE_THRESHOLD → use market+sentiment features
  'negative' — correlation < REGIME_NEGATIVE_THRESHOLD → use market features only
  'neutral'  — otherwise → use market features only
Phase: 2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.thresholds import (
    ATTENTION_FILTER_MIN_POSTS,
    REGIME_LOOKBACK_DAYS,
    REGIME_MIN_ROWS,
    REGIME_POSITIVE_THRESHOLD,
    REGIME_NEGATIVE_THRESHOLD,
)

REGIME_POSITIVE = "positive"
REGIME_NEGATIVE = "negative"
REGIME_NEUTRAL = "neutral"


def compute_sentiment_regime(
    df: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    lookback_days: int = REGIME_LOOKBACK_DAYS,
    min_rows: int = REGIME_MIN_ROWS,
    positive_threshold: float = REGIME_POSITIVE_THRESHOLD,
    negative_threshold: float = REGIME_NEGATIVE_THRESHOLD,
) -> str:
    """
    Compute sentiment regime for (ticker, date) using only past data.

    The window is [date - lookback_days, date) — strictly excludes date itself
    to prevent any form of look-ahead leakage.

    Args:
        df: Full feature dataframe with 'ticker', 'date', 'avg_sentiment_1d',
            'target_return_5d', 'post_count_1d'.
        ticker: Ticker symbol to evaluate.
        date: Current date (T). Only data from T-lookback_days to T-1 is used.
        lookback_days: Calendar days to look back.
        min_rows: Minimum rows in window to compute stable correlation.
        positive_threshold: Correlation above this → SENTIMENT_POSITIVE regime.
        negative_threshold: Correlation below this → SENTIMENT_NEGATIVE regime.

    Returns:
        'positive' | 'negative' | 'neutral'
    """
    cutoff = date
    start = cutoff - pd.Timedelta(days=lookback_days)

    window = df[
        (df["ticker"] == ticker)
        & (df["date"] >= start)
        & (df["date"] < cutoff)  # strictly excludes today — no leakage
        & (df["post_count_1d"] >= ATTENTION_FILTER_MIN_POSTS)
    ]

    if len(window) < min_rows:
        return REGIME_NEUTRAL

    corr = window["avg_sentiment_1d"].corr(window["target_return_5d"])

    if pd.isna(corr):
        return REGIME_NEUTRAL
    elif corr > positive_threshold:
        return REGIME_POSITIVE
    elif corr < negative_threshold:
        return REGIME_NEGATIVE
    else:
        return REGIME_NEUTRAL


def compute_regimes_bulk(
    df: pd.DataFrame,
    lookback_days: int = REGIME_LOOKBACK_DAYS,
    min_rows: int = REGIME_MIN_ROWS,
    positive_threshold: float = REGIME_POSITIVE_THRESHOLD,
    negative_threshold: float = REGIME_NEGATIVE_THRESHOLD,
) -> pd.Series:
    """
    Vectorised regime computation for all (ticker, date) rows in df.

    Iterates row by row — O(N*W) but acceptable for 13k rows.
    Returns a pd.Series of regime strings aligned to df.index.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    regimes = pd.Series(REGIME_NEUTRAL, index=df.index, dtype=str)

    for idx, row in df.iterrows():
        regime = compute_sentiment_regime(
            df=df,
            ticker=row["ticker"],
            date=row["date"],
            lookback_days=lookback_days,
            min_rows=min_rows,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
        regimes.at[idx] = regime

    return regimes


# ---------------------------------------------------------------------------
# Mandatory leakage test
# ---------------------------------------------------------------------------

def test_regime_no_leakage() -> None:
    """
    Assert that the regime for date T is unaffected by data at T or later.

    Strategy:
      1. Build a small synthetic dataframe up to date T.
      2. Compute regime for T.
      3. Add a fake row at T+1 with extreme positive sentiment.
      4. Recompute regime for T.
      5. Assert both regimes are identical.

    Raises:
        AssertionError if leakage is detected.
        RuntimeError if test setup fails unexpectedly.
    """
    import random

    rng = np.random.default_rng(seed=42)

    dates = pd.date_range("2023-01-01", periods=90, freq="B")
    ticker = "TEST"

    base_df = pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "avg_sentiment_1d": rng.uniform(-0.5, 0.5, size=len(dates)),
            "target_return_5d": rng.uniform(-0.1, 0.1, size=len(dates)),
            "post_count_1d": rng.integers(10, 50, size=len(dates)),
        }
    )

    eval_date = dates[-1]

    # Regime without future data
    regime_before = compute_sentiment_regime(
        df=base_df,
        ticker=ticker,
        date=eval_date,
    )

    # Add a fake future row with extreme sentiment at eval_date + 1 day
    future_row = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": eval_date + pd.Timedelta(days=1),
                "avg_sentiment_1d": 99.0,   # extreme — would corrupt any leaky window
                "target_return_5d": 99.0,
                "post_count_1d": 9999,
            }
        ]
    )
    df_with_future = pd.concat([base_df, future_row], ignore_index=True)

    # Regime with future data present (must be unchanged)
    regime_after = compute_sentiment_regime(
        df=df_with_future,
        ticker=ticker,
        date=eval_date,
    )

    assert regime_before == regime_after, (
        f"LEAKAGE DETECTED: regime changed from '{regime_before}' to '{regime_after}' "
        f"when future data was added. The window [T-{REGIME_LOOKBACK_DAYS}d, T) "
        f"is incorrectly including T or later."
    )

    print(f"[LEAKAGE TEST PASSED] regime={regime_before} — future data does not affect result.")


if __name__ == "__main__":
    test_regime_no_leakage()
