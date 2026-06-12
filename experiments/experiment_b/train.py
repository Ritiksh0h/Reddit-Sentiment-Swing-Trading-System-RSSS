#!/usr/bin/env python3
"""
Module: experiments/experiment_b/train.py
Purpose: Experiment B — Regime-Aware Sentiment Model.

Thesis: Sentiment works but only in the right regime. Detect the current regime
from the rolling correlation between sentiment and returns. Use sentiment features
only when the regime supports them.

Architecture:
  Step 1: Run mandatory leakage test on regime_detector — HARD GATE.
  Step 2: Compute rolling sentiment regime for every (ticker, date) row.
  Step 3: Train two models:
          - Model_B_positive: trained on rows where regime = 'positive'
            using MARKET + SENTIMENT features
          - Model_B_market: trained on ALL rows using MARKET features only
            (fallback for non-positive regime)
  Step 4: At inference, route each test row to the appropriate model.
  Step 5: Evaluate combined IC on test set and run backtest.

WARNING: This experiment has the highest overfitting risk. If B significantly
outperforms A and C on 2024, verify regime computation is truly using past data
only. The leakage test must pass before any training proceeds.

Usage:
    python experiments/experiment_b/train.py
    python experiments/experiment_b/train.py --debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import FEATURES_PARQUET
from config.thresholds import (
    ATTENTION_FILTER_MIN_POSTS,
    REGIME_LOOKBACK_DAYS,
    REGIME_MIN_ROWS,
    REGIME_POSITIVE_THRESHOLD,
    REGIME_NEGATIVE_THRESHOLD,
)
from experiments.experiment_b.regime_detector import (
    REGIME_POSITIVE,
    REGIME_NEGATIVE,
    REGIME_NEUTRAL,
    compute_regimes_bulk,
    test_regime_no_leakage,
)
from experiments.shared.backtest import run_backtest
from experiments.shared.trainer import train_xgboost, evaluate_ic, predict
from pipeline.feature_schema import MARKET_FEATURES, REDDIT_FEATURES, TARGET_COL
from utils.logger import get_logger

log = get_logger(__name__)

RESULTS_PATH = Path(__file__).parent / "results.json"
SPY_2024_RETURN = 0.2605  # actual SPY total return 2024

# Sentiment features used in positive-regime model (no volume/attention counts)
SENTIMENT_FEATURES: list[str] = [
    "avg_sentiment_1d",
    "avg_sentiment_3d",
    "avg_sentiment_hc",
    "weighted_sentiment",
    "bullish_ratio",
    "sentiment_accel",
    "sentiment_std",
]

POSITIVE_REGIME_FEATURES: list[str] = MARKET_FEATURES + SENTIMENT_FEATURES


def run_experiment_b(debug: bool = False) -> dict:
    """Train, evaluate, and backtest Experiment B. Returns results dict."""
    log.info("experiment_b_start")

    # ------------------------------------------------------------------
    # HARD GATE: leakage test must pass before any training
    # ------------------------------------------------------------------
    log.info("leakage_test_start")
    test_regime_no_leakage()
    log.info("leakage_test_passed")

    # ------------------------------------------------------------------
    # 1. Load features
    # ------------------------------------------------------------------
    df = pd.read_parquet(FEATURES_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    log.info("features_loaded", rows=len(df))

    if debug:
        # Keep all tickers but limit training to make iteration faster
        df = df[df["ticker"].isin(df["ticker"].unique()[:10])].copy()
        log.info("debug_mode", rows=len(df))

    # ------------------------------------------------------------------
    # 2. Compute rolling regime for every row
    #    This uses only past data via compute_sentiment_regime().
    # ------------------------------------------------------------------
    log.info("regime_computation_start", total_rows=len(df))
    df["regime"] = compute_regimes_bulk(
        df,
        lookback_days=REGIME_LOOKBACK_DAYS,
        min_rows=REGIME_MIN_ROWS,
        positive_threshold=REGIME_POSITIVE_THRESHOLD,
        negative_threshold=REGIME_NEGATIVE_THRESHOLD,
    )

    regime_dist = df["regime"].value_counts().to_dict()
    log.info("regime_distribution", **regime_dist)

    # ------------------------------------------------------------------
    # 3. Split into train / test
    # ------------------------------------------------------------------
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    log.info("split_sizes", train_rows=len(train), test_rows=len(test))

    # ------------------------------------------------------------------
    # 4. Train Model_B_positive (MARKET + SENTIMENT, positive-regime rows only)
    # ------------------------------------------------------------------
    train_positive = train[train["regime"] == REGIME_POSITIVE]
    log.info("positive_regime_train_rows", rows=len(train_positive))

    if len(train_positive) >= 30:
        log.info("training_model_b_positive", features=len(POSITIVE_REGIME_FEATURES))
        model_positive = train_xgboost(train_positive, POSITIVE_REGIME_FEATURES, TARGET_COL)
        model_positive_trained = True
    else:
        log.warning(
            "insufficient_positive_rows",
            rows=len(train_positive),
            fallback="market_model_only",
        )
        model_positive = None
        model_positive_trained = False

    # ------------------------------------------------------------------
    # 5. Train Model_B_market (MARKET features, all rows — fallback)
    # ------------------------------------------------------------------
    log.info("training_model_b_market", features=len(MARKET_FEATURES))
    model_market = train_xgboost(train, MARKET_FEATURES, TARGET_COL)
    log.info("model_b_market_trained")

    # ------------------------------------------------------------------
    # 6. Inference: route each test row to the appropriate model
    # ------------------------------------------------------------------
    test = test.copy()
    test["pred_return"] = np.nan

    test_positive = test[test["regime"] == REGIME_POSITIVE]
    test_other = test[test["regime"] != REGIME_POSITIVE]

    if model_positive_trained and len(test_positive) > 0:
        test.loc[test_positive.index, "pred_return"] = predict(
            model_positive, test_positive, POSITIVE_REGIME_FEATURES
        )
        log.info("positive_regime_predictions", rows=len(test_positive))

    # Fallback: market model for non-positive rows AND any positive rows
    # if model_positive was not trained
    remaining_mask = test["pred_return"].isna()
    if remaining_mask.any():
        test.loc[remaining_mask, "pred_return"] = predict(
            model_market, test[remaining_mask], MARKET_FEATURES
        )

    # ------------------------------------------------------------------
    # 7. Compute combined IC on test set
    # ------------------------------------------------------------------
    ic_test, _ = spearmanr(test["pred_return"], test[TARGET_COL])
    ic_test = float(ic_test) if np.isfinite(ic_test) else 0.0
    log.info("combined_ic", ic_test=round(ic_test, 4))

    # IC split by regime
    ic_by_regime: dict[str, float] = {}
    for regime in [REGIME_POSITIVE, REGIME_NEGATIVE, REGIME_NEUTRAL]:
        subset = test[test["regime"] == regime]
        if len(subset) >= 10:
            c, _ = spearmanr(subset["pred_return"], subset[TARGET_COL])
            ic_by_regime[regime] = round(float(c) if np.isfinite(c) else 0.0, 4)
        else:
            ic_by_regime[regime] = None

    # ------------------------------------------------------------------
    # 8. Backtest using the combined prediction column
    # ------------------------------------------------------------------
    # Inject predictions into a column backtest engine will use
    # Backtest engine reads model.predict() — we wrap predictions in a
    # thin object so the shared backtest can call model.predict(X)
    class _WrappedPreds:
        """Thin wrapper so run_backtest can call .predict() with the precomputed values."""
        def __init__(self, preds: np.ndarray):
            self._preds = preds

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            # X is a subset of test — map by positional index won't work.
            # We pre-attach predictions to test_df before passing to backtest,
            # so this should not be called. Safety fallback: return zeros.
            return np.zeros(len(X))

    # We handle routing manually above, so inject pred_return directly.
    # Override run_backtest to accept pre-computed predictions via a
    # special column rather than calling model.predict() again.
    log.info("backtest_start")
    backtest_results = _run_backtest_with_precomputed_preds(
        test_df=test,
        spy_return=SPY_2024_RETURN,
    )
    log.info(
        "backtest_complete",
        sharpe=round(backtest_results["sharpe_ratio"], 3),
        total_return=round(backtest_results["total_return"], 4),
        n_trades=backtest_results["n_trades"],
        beats_spy=backtest_results["beats_spy"],
    )

    # ------------------------------------------------------------------
    # 9. Assemble results
    # ------------------------------------------------------------------
    results = {
        "experiment": "B",
        "thesis": "regime_aware_sentiment_model",
        "regime_distribution_train": {
            k: int(v) for k, v in train["regime"].value_counts().items()
        },
        "regime_distribution_test": {
            k: int(v) for k, v in test["regime"].value_counts().items()
        },
        "model_positive_trained": model_positive_trained,
        "positive_regime_features": POSITIVE_REGIME_FEATURES,
        "market_features": MARKET_FEATURES,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_rows": len(train_positive),
        "ic_test": round(ic_test, 6),
        "ic_by_regime": ic_by_regime,
        "passes_ic_threshold": ic_test > 0.05,
        "passes_sharpe_threshold": backtest_results["sharpe_ratio"] > 1.0,
        "regime_params": {
            "lookback_days": REGIME_LOOKBACK_DAYS,
            "min_rows": REGIME_MIN_ROWS,
            "positive_threshold": REGIME_POSITIVE_THRESHOLD,
            "negative_threshold": REGIME_NEGATIVE_THRESHOLD,
        },
        **{k: v for k, v in backtest_results.items() if k not in ("equity_curve", "trades")},
        "trade_log": backtest_results.get("trades", []),
        "equity_curve": backtest_results.get("equity_curve", []),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("results_saved", path=str(RESULTS_PATH))
    _print_summary(results)
    return results


def _run_backtest_with_precomputed_preds(
    test_df: pd.DataFrame,
    spy_return: float,
    starting_capital: float = 1_000.0,
    max_positions: int = 3,
    hold_days: int = 5,
    slippage: float = 0.001,
    fee_per_leg: float = 0.0005,
    min_pred_return: float = 0.01,
) -> dict:
    """
    Backtest using precomputed 'pred_return' column in test_df.
    Identical mechanics to shared/backtest.py — reproduces logic here
    because predictions are already computed via regime routing.
    Includes Fix 1 (cooldown), Fix 2 (last valid close), Fix 3 (0.01 threshold).
    """
    from experiments.shared.backtest import _last_valid_close
    from experiments.shared.metrics import compute_metrics

    df = test_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    trading_days = sorted(df["date"].unique())
    cash = starting_capital
    open_positions: list[dict] = []
    equity_curve: list[float] = [starting_capital]
    trades: list[dict] = []
    recently_closed: dict = {}
    cooldown_window = pd.Timedelta(days=hold_days * 1.4)

    for day in trading_days:
        today_df = df[df["date"] == day]

        # Close matured positions
        still_open = []
        for pos in open_positions:
            days_held = (day - pos["entry_date"]).days
            if days_held >= hold_days * 1.4:
                ticker_rows = today_df[today_df["ticker"] == pos["ticker"]]
                # Fix 2: use last valid close if today's price is missing
                if len(ticker_rows) > 0:
                    exit_close = float(ticker_rows["close"].iloc[0])
                else:
                    exit_close = _last_valid_close(df, pos["ticker"], day)
                    if exit_close is None:
                        still_open.append(pos)
                        continue
                exit_price = exit_close * (1 - slippage)
                gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                fee = exit_price * pos["shares"] * fee_per_leg
                net_pnl = gross_pnl - fee
                cash += pos["entry_price"] * pos["shares"] + net_pnl
                recently_closed[pos["ticker"]] = day  # Fix 1
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(day.date()),
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(exit_price, 4),
                    "shares": pos["shares"],
                    "pred_return": round(pos["pred_return"], 6),
                    "regime": pos.get("regime", "unknown"),
                    "gross_pnl": round(gross_pnl, 4),
                    "pnl": round(net_pnl, 4),
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open new positions
        n_open = len(open_positions)
        slots = max_positions - n_open
        if slots > 0 and len(today_df) > 0:
            candidates = today_df[today_df["pred_return"] >= min_pred_return].copy()
            # Fix 1: exclude tickers currently held or in cooldown
            open_tickers = {p["ticker"] for p in open_positions}
            candidates = candidates[
                ~candidates["ticker"].isin(open_tickers)
                & ~candidates["ticker"].apply(
                    lambda t: t in recently_closed
                    and (day - recently_closed[t]) < cooldown_window
                )
            ]
            candidates = candidates.nlargest(slots, "pred_return")
            for _, row in candidates.iterrows():
                if cash <= 0:
                    break
                position_value = min(starting_capital / max_positions, cash)
                entry_price = float(row["close"]) * (1 + slippage)
                fee_entry = entry_price * (position_value / entry_price) * fee_per_leg
                shares = (position_value - fee_entry) / entry_price
                if shares <= 0:
                    continue
                cash -= shares * entry_price + fee_entry
                open_positions.append({
                    "ticker": row["ticker"],
                    "entry_date": day,
                    "entry_price": entry_price,
                    "shares": shares,
                    "pred_return": float(row["pred_return"]),
                    "regime": row.get("regime", "unknown"),
                })

        # Mark to market
        open_value = 0.0
        for pos in open_positions:
            ticker_rows = today_df[today_df["ticker"] == pos["ticker"]]
            mtm_price = float(ticker_rows["close"].iloc[0]) if len(ticker_rows) > 0 else pos["entry_price"]
            open_value += pos["shares"] * mtm_price
        equity_curve.append(cash + open_value)

    # Force-close remainders
    last_day = trading_days[-1]
    last_df = df[df["date"] == last_day]
    for pos in open_positions:
        ticker_rows = last_df[last_df["ticker"] == pos["ticker"]]
        # Fix 2: use last valid close; skip if truly no price data
        if len(ticker_rows) > 0:
            exit_close = float(ticker_rows["close"].iloc[0])
        else:
            exit_close = _last_valid_close(df, pos["ticker"], last_day + pd.Timedelta(days=1))
            if exit_close is None:
                continue
        exit_price = exit_close * (1 - slippage)
        gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        fee = exit_price * pos["shares"] * fee_per_leg
        net_pnl = gross_pnl - fee
        trades.append({
            "ticker": pos["ticker"],
            "entry_date": str(pos["entry_date"].date()),
            "exit_date": str(last_day.date()),
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(exit_price, 4),
            "shares": pos["shares"],
            "pred_return": round(pos["pred_return"], 6),
            "regime": pos.get("regime", "unknown"),
            "gross_pnl": round(gross_pnl, 4),
            "pnl": round(net_pnl, 4),
        })

    metrics = compute_metrics(equity_curve, trades, spy_return)
    metrics["equity_curve"] = [round(v, 4) for v in equity_curve]
    metrics["trades"] = trades
    return metrics


def _print_summary(r: dict) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT B RESULTS")
    print("=" * 60)
    print(f"  Thesis:         {r['thesis']}")
    print(f"  Train rows:     {r['train_rows']}")
    print(f"  Test rows:      {r['test_rows']}")
    print(f"  Positive-regime train rows: {r['train_positive_rows']}")
    print(f"  Regime dist (test): {r['regime_distribution_test']}")
    print(f"  IC (test):      {r['ic_test']:.4f}  {'PASS' if r['passes_ic_threshold'] else 'FAIL'} (threshold: 0.05)")
    print(f"  IC by regime:   {r['ic_by_regime']}")
    print(f"  Sharpe:         {r['sharpe_ratio']:.3f}  {'PASS' if r['passes_sharpe_threshold'] else 'FAIL'} (threshold: 1.0)")
    print(f"  Total return:   {r['total_return']*100:.1f}%")
    print(f"  Annual return:  {r['annualized_return']*100:.1f}%")
    print(f"  Max drawdown:   {r['max_drawdown']*100:.1f}%")
    print(f"  Win rate:       {r['win_rate']*100:.1f}%")
    print(f"  N trades:       {r['n_trades']}")
    print(f"  SPY return:     {r['spy_return']*100:.1f}%")
    print(f"  Beats SPY:      {r['beats_spy']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment B")
    parser.add_argument("--debug", action="store_true", help="Use small data subset")
    args = parser.parse_args()
    run_experiment_b(debug=args.debug)
