"""
Module: tests/test_backtest.py
Purpose: Unit tests for experiments/shared/backtest.py.
         Covers the three bugs fixed in Phase 2B: ticker concentration,
         missing exit prices, and threshold filtering.
Phase: 2B
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.shared.backtest import run_backtest, HOLD_DAYS


# ---------------------------------------------------------------------------
# Minimal stub model — always predicts a fixed value
# ---------------------------------------------------------------------------

class _ConstantModel:
    def __init__(self, pred: float = 0.05):
        self._pred = pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._pred)


class _PerRowModel:
    """Predicts a value from a pre-built mapping keyed by (ticker, date string)."""
    def __init__(self, preds: dict):
        self._preds = preds  # {(ticker, date_str): pred_return}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # X is a slice of the test_df — use its index to find the parent rows
        raise NotImplementedError("Use _ConstantModel or inject pred_return directly")


# ---------------------------------------------------------------------------
# Shared test dataframe factory
# ---------------------------------------------------------------------------

def _make_test_df(
    tickers: list[str],
    n_days: int = 30,
    base_price: float = 100.0,
    daily_return: float = 0.005,
    post_count_1d: float = 15.0,
    mention_growth_7d: float = 0.5,
) -> pd.DataFrame:
    """
    Build a minimal test dataframe that satisfies backtest engine requirements.
    All feature columns are filled with benign constants.
    """
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows = []
    for ticker in tickers:
        price = base_price
        for date in dates:
            price = price * (1 + daily_return)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "close": round(price, 4),
                    "volume": 1_000_000,
                    "returns_1d": daily_return,
                    "returns_5d": daily_return * 5,
                    "returns_20d": daily_return * 20,
                    "rsi_14": 55.0,
                    "atr_14": price * 0.01,
                    "relative_volume": 1.5,
                    "dist_from_20ma": 0.01,
                    "dist_from_50ma": 0.02,
                    "post_count_1d": post_count_1d,
                    "mention_growth_7d": mention_growth_7d,
                    "avg_sentiment_1d": 0.3,
                    "avg_sentiment_3d": 0.25,
                    "avg_sentiment_hc": 0.35,
                    "weighted_sentiment": 0.28,
                    "bullish_ratio": 0.6,
                    "sentiment_accel": 0.05,
                    "sentiment_std": 0.1,
                    "target_return_5d": daily_return * 5,
                    "target_return_10d": daily_return * 10,
                    "split": "test",
                }
            )
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "returns_1d", "returns_5d", "returns_20d",
    "rsi_14", "atr_14", "relative_volume",
    "dist_from_20ma", "dist_from_50ma",
    "close", "volume",
]


# ---------------------------------------------------------------------------
# Test 1: No duplicate ticker positions
# ---------------------------------------------------------------------------

def test_no_duplicate_ticker_positions():
    """
    Portfolio should never hold two positions in the same ticker simultaneously.

    Setup: one ticker across many days with always-bullish predictions.
    Even with 3 slots open and many opportunities, at most 1 position per ticker
    should be held at any time.
    """
    df = _make_test_df(tickers=["TSLA"], n_days=60)
    model = _ConstantModel(pred=0.05)

    result = run_backtest(
        model=model,
        test_df=df,
        feature_cols=FEATURE_COLS,
        spy_return=0.26,
    )

    trades = result["trades"]
    # Build a timeline of open positions and check for overlaps
    open_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_ = pd.Timestamp(t["exit_date"])
        open_windows.append((entry, exit_))

    # No two windows for the same ticker should overlap
    for i, (e1, x1) in enumerate(open_windows):
        for j, (e2, x2) in enumerate(open_windows):
            if i >= j:
                continue
            # Overlap exists if one starts before the other ends
            overlap = e1 < x2 and e2 < x1
            assert not overlap, (
                f"Duplicate TSLA position detected: "
                f"trade {i} [{e1}–{x1}] overlaps trade {j} [{e2}–{x2}]"
            )


# ---------------------------------------------------------------------------
# Test 2: Missing exit price handled — no zero-return trades
# ---------------------------------------------------------------------------

def test_missing_exit_price_handled():
    """
    A trade whose intended exit day has no price in the dataset should NOT
    produce a zero-return trade (entry_price == exit_price).

    Setup: create a ticker whose close is missing on the expected exit day.
    The backtest should either skip that trade or use the last valid prior price.
    """
    df = _make_test_df(tickers=["SNAP"], n_days=30)

    # Remove close data on day 8 (which falls within the hold window for a
    # trade entered on day 1–2)
    day8 = df["date"].sort_values().unique()[7]
    df = df[~((df["ticker"] == "SNAP") & (df["date"] == day8))].copy()

    model = _ConstantModel(pred=0.05)
    result = run_backtest(
        model=model,
        test_df=df,
        feature_cols=FEATURE_COLS,
        spy_return=0.26,
    )

    trades = result["trades"]
    zero_return_trades = [
        t for t in trades
        if abs(t["entry_price"] - t["exit_price"]) < 1e-6
    ]
    assert len(zero_return_trades) == 0, (
        f"Found {len(zero_return_trades)} zero-return trade(s) — "
        f"missing price fallback did not work: {zero_return_trades}"
    )


# ---------------------------------------------------------------------------
# Test 3: min_pred_return threshold filtering
# ---------------------------------------------------------------------------

def test_min_pred_return_threshold():
    """
    Signals with predicted return below min_pred_return must not generate trades.
    Signals at or above the threshold must be tradeable.
    """
    # Use two tickers: one that barely misses the threshold, one that passes
    df_below = _make_test_df(tickers=["LOW_TICKER"], n_days=20)
    df_above = _make_test_df(tickers=["HIGH_TICKER"], n_days=20)

    class _SplitModel:
        """Predicts 0.005 for LOW_TICKER rows and 0.03 for HIGH_TICKER rows."""
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            # X is a slice of the dataframe — we can't access ticker here.
            # Instead, set different close prices and use a ConstantModel per df.
            raise NotImplementedError

    # Test below threshold: model predicts 0.005 < 0.01 (MIN_PRED_RETURN)
    model_below = _ConstantModel(pred=0.005)
    result_below = run_backtest(
        model=model_below,
        test_df=df_below,
        feature_cols=FEATURE_COLS,
        min_pred_return=0.01,
        spy_return=0.26,
    )
    assert result_below["n_trades"] == 0, (
        f"Expected 0 trades below threshold, got {result_below['n_trades']}"
    )

    # Test above threshold: model predicts 0.03 > 0.01
    model_above = _ConstantModel(pred=0.03)
    result_above = run_backtest(
        model=model_above,
        test_df=df_above,
        feature_cols=FEATURE_COLS,
        min_pred_return=0.01,
        spy_return=0.26,
    )
    assert result_above["n_trades"] > 0, (
        f"Expected trades above threshold, got {result_above['n_trades']}"
    )


# ---------------------------------------------------------------------------
# Test 4: Concentration guard — single ticker < 50% of all trades
# ---------------------------------------------------------------------------

def test_ticker_concentration_bounded():
    """
    When multiple tickers are equally ranked, no single ticker should
    dominate the trade log when there are alternatives available.
    """
    # 5 tickers, all with identical features and model predictions
    tickers = ["TSLA", "AMD", "NVDA", "META", "PLTR"]
    df = _make_test_df(tickers=tickers, n_days=60)
    model = _ConstantModel(pred=0.05)

    result = run_backtest(
        model=model,
        test_df=df,
        feature_cols=FEATURE_COLS,
        spy_return=0.26,
    )

    if result["n_trades"] == 0:
        pytest.skip("No trades generated — cannot assess concentration")

    trades = result["trades"]
    ticker_counts = {}
    for t in trades:
        ticker_counts[t["ticker"]] = ticker_counts.get(t["ticker"], 0) + 1

    max_count = max(ticker_counts.values())
    max_pct = max_count / len(trades)

    assert max_pct < 0.5, (
        f"Single ticker dominates trade log: "
        f"{max(ticker_counts, key=ticker_counts.get)} = {max_pct:.1%} of trades"
    )


# ---------------------------------------------------------------------------
# Test 5: Take-profit cap closes position early
# ---------------------------------------------------------------------------

def test_take_profit_cap_triggers():
    """
    A position that gains >= TAKE_PROFIT_CAP should close early with
    exit_reason='take_profit_cap', not wait for hold_days.

    Setup: price rises 20% on day 3 (well above 15% cap).
    Assert: trade exits on day 3, not day 5+.
    Assert: exit_reason == 'take_profit_cap'.
    """
    from config.thresholds import TAKE_PROFIT_CAP

    # Build a single-ticker dataframe with a big jump on day 3
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    prices = [100.0] * 20
    prices[2] = 121.0   # day 3: +21% from entry on day 1 → exceeds 15% cap

    rows = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        rows.append(
            {
                "ticker": "JUMP",
                "date": date,
                "close": price,
                "volume": 1_000_000,
                "returns_1d": 0.0,
                "returns_5d": 0.0,
                "returns_20d": 0.0,
                "rsi_14": 55.0,
                "atr_14": 1.0,
                "relative_volume": 1.5,
                "dist_from_20ma": 0.0,
                "dist_from_50ma": 0.0,
                "post_count_1d": 15.0,
                "mention_growth_7d": 0.5,
                "avg_sentiment_1d": 0.3,
                "avg_sentiment_3d": 0.25,
                "avg_sentiment_hc": 0.35,
                "weighted_sentiment": 0.28,
                "bullish_ratio": 0.6,
                "sentiment_accel": 0.05,
                "sentiment_std": 0.1,
                "target_return_5d": 0.05,
                "target_return_10d": 0.10,
                "split": "test",
            }
        )

    df = pd.DataFrame(rows)
    model = _ConstantModel(pred=0.05)

    result = run_backtest(
        model=model,
        test_df=df,
        feature_cols=FEATURE_COLS,
        spy_return=0.26,
    )

    trades = result["trades"]
    assert len(trades) > 0, "No trades generated — cannot test take-profit"

    tp_trades = [t for t in trades if t.get("exit_reason") == "take_profit_cap"]
    assert len(tp_trades) > 0, (
        f"Expected at least one take_profit_cap exit. "
        f"Exit reasons seen: {[t.get('exit_reason') for t in trades]}"
    )

    # The take-profit trade must exit on or before day 3 (2024-01-04 = 3rd business day)
    day3 = pd.Timestamp("2024-01-04").date()
    for t in tp_trades:
        exit_date = pd.Timestamp(t["exit_date"]).date()
        assert exit_date <= day3, (
            f"Take-profit should have triggered by {day3}, "
            f"but exited on {exit_date}"
        )
        # Exit price should reflect the 21% jump (≈ 121 * (1 - slippage))
        assert t["exit_price"] > 118.0, (
            f"Expected exit near 121, got {t['exit_price']}"
        )
