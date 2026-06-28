"""
test_leakage_checks.py — Unit tests for scripts/leakage_checks.py

3 tests:
  1. Inject a future-timestamp leak → CHECK 1 must FAIL
  2. Inject a duplicate (ticker, date) row → CHECK 6 must FAIL
  3. Run all checks on clean features_v2.parquet → OVERALL PASS
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.leakage_checks import (
    check_1_no_future_timestamps,
    check_6_duplicate_rows,
    run_all,
    FEATURE_COLS,
)

# ── Shared minimal dataframe fixture ─────────────────────────────────────────
@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """A small dataframe with two tickers and no integrity issues."""
    today = pd.Timestamp.today().normalize()
    old   = today - pd.Timedelta(days=60)

    rows = []
    for ticker in ("NVDA", "TSLA"):
        for offset in range(25):
            rows.append({
                "ticker":           ticker,
                "date":             old + pd.Timedelta(days=offset),
                "close":            100.0 + offset,
                "returns_1d":       0.01,
                "returns_20d":      0.05,
                "rsi_14":           55.0,
                "relative_volume":  1.2,
                "volume":           1_000_000.0,
                "post_count_1d":    10.0,
                "target_return_5d": 0.02,
                "target_return_1d": 0.01,
                "target_return_3d": 0.015,
                "sentiment_accel":  0.0,
                **{c: 0.0 for c in FEATURE_COLS
                   if c not in ("returns_1d", "returns_20d", "rsi_14",
                                "relative_volume", "volume", "post_count_1d",
                                "sentiment_accel")},
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Test 1: future-timestamp injection fails CHECK 1 ─────────────────────────
def test_check1_fails_on_future_timestamp_leak(minimal_df):
    """
    Inject a row where date is 2 days ago AND target_return_5d is filled.
    target_return_5d requires t+5 close price which does not exist yet.
    CHECK 1 must return FAIL.
    """
    today   = pd.Timestamp.today().normalize()
    bad_row = {
        "ticker":           "INJECTED",
        "date":             today - pd.Timedelta(days=2),
        "close":            50.0,
        "returns_1d":       0.01,
        "returns_20d":      0.02,
        "rsi_14":           50.0,
        "relative_volume":  1.0,
        "volume":           500_000.0,
        "post_count_1d":    5.0,
        "target_return_5d": 0.03,   # ← filled, but date is 2 days ago
        "target_return_1d": 0.01,
        "target_return_3d": 0.02,
        "sentiment_accel":  0.0,
        **{c: 0.0 for c in FEATURE_COLS
           if c not in ("returns_1d", "returns_20d", "rsi_14",
                        "relative_volume", "volume", "post_count_1d",
                        "sentiment_accel")},
    }
    df_bad = pd.concat(
        [minimal_df, pd.DataFrame([bad_row])], ignore_index=True
    )
    df_bad["date"] = pd.to_datetime(df_bad["date"])

    status, msgs = check_1_no_future_timestamps(df_bad)
    assert status == "FAIL", (
        f"Expected FAIL but got {status}. Messages: {msgs}"
    )
    assert any("INJECTED" in m for m in msgs), (
        "Expected INJECTED ticker to appear in violation messages"
    )


# ── Test 2: duplicate row injection fails CHECK 6 ────────────────────────────
def test_check6_fails_on_duplicate_row(minimal_df):
    """
    Duplicate the first row — creates an exact (ticker, date) collision.
    CHECK 6 must return FAIL.
    """
    dup_row = minimal_df.iloc[[0]].copy()
    df_dup  = pd.concat([minimal_df, dup_row], ignore_index=True)
    df_dup["date"] = pd.to_datetime(df_dup["date"])

    status, msgs = check_6_duplicate_rows(df_dup)
    assert status == "FAIL", (
        f"Expected FAIL but got {status}. Messages: {msgs}"
    )
    assert any("duplicate" in m.lower() for m in msgs), (
        "Expected duplicate count in messages"
    )


# ── Test 3: clean parquet passes all checks ───────────────────────────────────
def test_all_checks_pass_on_clean_data():
    """
    Run the full leakage suite on features_v2.parquet.
    OVERALL must be PASS (WARNs are allowed, FAILs are not).

    This test is skipped if the parquet doesn't exist (e.g. CI runner
    without the data file).
    """
    parquet = Path(__file__).resolve().parent.parent / \
              "data" / "features" / "features_v2.parquet"
    if not parquet.exists():
        pytest.skip("features_v2.parquet not found — skipping clean-data test")

    passed = run_all(parquet)
    assert passed, "Leakage suite reported OVERALL FAIL on clean features_v2.parquet"
