#!/usr/bin/env python3
"""
Module: experiments/experiment_c/train_10d.py
Purpose: Experiment C variant — 10-day hold period.

Same architecture as train.py (expanded dataset + combined model) but trained
to predict target_return_10d instead of target_return_5d, and backtested with
a 10-day hold period.

Hypothesis: Social sentiment signals have stronger predictive power at 10-20
day horizons. Sentiment takes time to propagate through retail investor
behavior into price.

Do NOT modify train.py — this is an additive comparison only.

Usage:
    python experiments/experiment_c/train_10d.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.thresholds import (
    ATTENTION_FILTER_MIN_POSTS,
    EXPANDED_PARQUET_PATH,
    EXPANDED_FEATURES_PATH,
)
from experiments.shared.backtest import run_backtest
from experiments.shared.trainer import train_xgboost, evaluate_ic
from pipeline.feature_schema import MARKET_FEATURES, TARGET_COL_10D
from utils.logger import get_logger

log = get_logger(__name__)

RESULTS_PATH_10D = Path(__file__).parent / "results_10d.json"
SPY_2024_RETURN = 0.2605

COMBINED_FEATURES = MARKET_FEATURES + [
    "avg_sentiment_1d",
    "avg_sentiment_3d",
    "avg_sentiment_hc",
    "weighted_sentiment",
    "bullish_ratio",
    "sentiment_accel",
    "sentiment_std",
]


def run_experiment_c_10d() -> dict:
    """Train and backtest Experiment C with 10-day hold. Returns results dict."""
    if not os.path.exists(EXPANDED_PARQUET_PATH):
        print(f"\nEXPERIMENT C 10D: WAITING — {EXPANDED_PARQUET_PATH} not found.")
        sys.exit(0)

    if not os.path.exists(EXPANDED_FEATURES_PATH):
        print("\nExpanded features not built yet. Run train.py first.")
        sys.exit(1)

    log.info("experiment_c_10d_start")

    # ------------------------------------------------------------------
    # 1. Load expanded features
    # ------------------------------------------------------------------
    df = pd.read_parquet(EXPANDED_FEATURES_PATH)
    log.info("features_loaded", rows=len(df))

    if "target_return_10d" not in df.columns:
        print("\nERROR: target_return_10d not found in features.")
        print("Run: python pipeline/01_feature_builder.py --force-recompute")
        print("Then delete data/features/features_expanded.parquet and rerun train.py")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Density filter
    # ------------------------------------------------------------------
    df_filtered = df[df["post_count_1d"] >= ATTENTION_FILTER_MIN_POSTS].copy()
    pct_retained = len(df_filtered) / max(len(df), 1)
    log.info(
        "density_filter_applied",
        total_rows=len(df),
        filtered_rows=len(df_filtered),
        pct_retained=round(pct_retained, 4),
    )

    # ------------------------------------------------------------------
    # 3. Drop rows where 10d target is NaN (last 10 trading days of data)
    # ------------------------------------------------------------------
    df_filtered = df_filtered[df_filtered["target_return_10d"].notna()].copy()
    log.info("nan_target_dropped", remaining=len(df_filtered))

    # ------------------------------------------------------------------
    # 4. Train / test split
    # ------------------------------------------------------------------
    train = df_filtered[df_filtered["split"] == "train"].copy()
    test = df_filtered[df_filtered["split"] == "test"].copy()

    log.info("split_sizes", train_rows=len(train), test_rows=len(test))

    if len(train) < 50:
        log.error("insufficient_train_rows", rows=len(train))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Train model on 10d target
    # ------------------------------------------------------------------
    log.info("training_model_10d", features=len(COMBINED_FEATURES))
    model = train_xgboost(train, COMBINED_FEATURES, TARGET_COL_10D)
    log.info("model_trained")

    # ------------------------------------------------------------------
    # 6. Evaluate IC on 10d target
    # ------------------------------------------------------------------
    ic_test = evaluate_ic(model, test, COMBINED_FEATURES, TARGET_COL_10D)
    log.info("ic_evaluated", ic_test=round(ic_test, 4))

    # ------------------------------------------------------------------
    # 7. Backtest with 10-day hold period
    # ------------------------------------------------------------------
    log.info("backtest_start", hold_days=10)
    backtest_results = run_backtest(
        model=model,
        test_df=test,
        feature_cols=COMBINED_FEATURES,
        hold_days=10,
        spy_return=SPY_2024_RETURN,
    )
    log.info(
        "backtest_complete",
        sharpe=round(backtest_results["sharpe_ratio"], 3),
        total_return=round(backtest_results["total_return"], 4),
        n_trades=backtest_results["n_trades"],
    )

    # ------------------------------------------------------------------
    # 8. Assemble results
    # ------------------------------------------------------------------
    results = {
        "experiment": "C_10d",
        "thesis": "expanded_dataset_combined_model_10d_hold",
        "target_col": TARGET_COL_10D,
        "hold_days": 10,
        "features": COMBINED_FEATURES,
        "density_filter_min_posts": ATTENTION_FILTER_MIN_POSTS,
        "train_rows": len(train),
        "test_rows": len(test),
        "ic_test": round(ic_test, 6),
        "passes_ic_threshold": ic_test > 0.05,
        "passes_sharpe_threshold": backtest_results["sharpe_ratio"] > 1.0,
        **{k: v for k, v in backtest_results.items() if k not in ("equity_curve", "trades")},
        "trade_log": backtest_results.get("trades", []),
        "equity_curve": backtest_results.get("equity_curve", []),
    }

    RESULTS_PATH_10D.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH_10D, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("results_saved", path=str(RESULTS_PATH_10D))
    _print_summary(results)
    return results


def _print_summary(r: dict) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT C — 10-DAY HOLD RESULTS")
    print("=" * 60)
    print(f"  Target:         {r['target_col']}")
    print(f"  Hold period:    {r['hold_days']} days")
    print(f"  Train rows:     {r['train_rows']} (post density filter)")
    print(f"  Test rows:      {r['test_rows']} (post density filter)")
    print(f"  IC (test):      {r['ic_test']:.4f}  {'PASS' if r['passes_ic_threshold'] else 'FAIL'} (threshold: 0.05)")
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
    run_experiment_c_10d()
