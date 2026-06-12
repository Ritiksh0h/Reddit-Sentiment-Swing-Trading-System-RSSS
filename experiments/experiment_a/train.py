#!/usr/bin/env python3
"""
Module: experiments/experiment_a/train.py
Purpose: Experiment A — Attention Filter + Market Model.

Thesis: Reddit identifies which stocks have crowd attention. Market features
predict direction. Keep these roles separate.

Architecture:
  Step 1: Apply daily attention filter (post_count_1d >= 10, mention_growth_7d >= 0.3)
  Step 2: Train market-only XGBoost on filtered universe (2019-2023)
  Step 3: Evaluate IC on filtered 2024 test set
  Step 4: Run backtest and save results

The attention filter is a PRE-SELECTION GATE — not a model feature.

Usage:
    python experiments/experiment_a/train.py
    python experiments/experiment_a/train.py --debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import FEATURES_PARQUET
from config.thresholds import (
    ATTENTION_FILTER_MIN_POSTS,
    ATTENTION_FILTER_MIN_GROWTH,
)
from experiments.shared.backtest import run_backtest
from experiments.shared.trainer import train_xgboost, evaluate_ic
from pipeline.feature_schema import MARKET_FEATURES, TARGET_COL
from utils.logger import get_logger

log = get_logger(__name__)

RESULTS_PATH = Path(__file__).parent / "results.json"
SPY_2024_RETURN = 0.2605  # actual SPY total return 2024


def attention_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the attention gate. Returns only rows passing both thresholds."""
    return df[
        (df["post_count_1d"] >= ATTENTION_FILTER_MIN_POSTS)
        & (df["mention_growth_7d"] >= ATTENTION_FILTER_MIN_GROWTH)
    ].copy()


def run_experiment_a(debug: bool = False) -> dict:
    """Train, evaluate, and backtest Experiment A. Returns results dict."""
    log.info("experiment_a_start", features=MARKET_FEATURES, target=TARGET_COL)

    # ------------------------------------------------------------------
    # 1. Load features
    # ------------------------------------------------------------------
    df = pd.read_parquet(FEATURES_PARQUET)
    log.info("features_loaded", rows=len(df), cols=list(df.columns))

    # ------------------------------------------------------------------
    # 2. Apply attention filter
    # ------------------------------------------------------------------
    df_filtered = attention_filter(df)

    pct_retained = len(df_filtered) / max(len(df), 1)
    log.info(
        "attention_filter_applied",
        total_rows=len(df),
        filtered_rows=len(df_filtered),
        pct_retained=round(pct_retained, 4),
        min_posts=ATTENTION_FILTER_MIN_POSTS,
        min_growth=ATTENTION_FILTER_MIN_GROWTH,
    )

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    train = df_filtered[df_filtered["split"] == "train"].copy()
    test = df_filtered[df_filtered["split"] == "test"].copy()

    log.info("split_sizes", train_rows=len(train), test_rows=len(test))

    if len(train) < 50:
        log.error("insufficient_train_rows", rows=len(train))
        sys.exit(1)

    if len(test) < 10:
        log.error("insufficient_test_rows", rows=len(test))
        sys.exit(1)

    if debug:
        train = train.head(200)
        test = test.head(50)
        log.info("debug_mode", train_rows=len(train), test_rows=len(test))

    # ------------------------------------------------------------------
    # 4. Train market-only model on filtered universe
    # ------------------------------------------------------------------
    log.info("training_model", feature_count=len(MARKET_FEATURES))
    model = train_xgboost(train, MARKET_FEATURES, TARGET_COL)
    log.info("model_trained")

    # ------------------------------------------------------------------
    # 5. Evaluate IC on filtered test set
    # ------------------------------------------------------------------
    ic_test = evaluate_ic(model, test, MARKET_FEATURES, TARGET_COL)
    log.info("ic_evaluated", ic_test=round(ic_test, 4))

    # ------------------------------------------------------------------
    # 6. Backtest on filtered test set
    #    The pre_filter_fn is NOT applied here — test is already filtered.
    #    The filter was applied before splitting (pre-selection gate).
    # ------------------------------------------------------------------
    log.info("backtest_start")
    backtest_results = run_backtest(
        model=model,
        test_df=test,
        feature_cols=MARKET_FEATURES,
        pre_filter_fn=None,  # already filtered
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
    # 7. Assemble results
    # ------------------------------------------------------------------
    results = {
        "experiment": "A",
        "thesis": "attention_filter_plus_market_model",
        "filter_applied": True,
        "min_posts": ATTENTION_FILTER_MIN_POSTS,
        "min_growth": ATTENTION_FILTER_MIN_GROWTH,
        "feature_set": "market_only",
        "features": MARKET_FEATURES,
        "train_rows": len(train),
        "test_rows": len(test),
        "total_rows_before_filter": len(df),
        "pct_rows_retained": round(pct_retained, 4),
        "ic_test": round(ic_test, 6),
        "passes_ic_threshold": ic_test > 0.05,
        "passes_sharpe_threshold": backtest_results["sharpe_ratio"] > 1.0,
        **{k: v for k, v in backtest_results.items() if k not in ("equity_curve", "trades")},
        "trade_log": backtest_results.get("trades", []),
        "equity_curve": backtest_results.get("equity_curve", []),
    }

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("results_saved", path=str(RESULTS_PATH))

    _print_summary(results)
    return results


def _print_summary(r: dict) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT A RESULTS")
    print("=" * 60)
    print(f"  Thesis:         {r['thesis']}")
    print(f"  Train rows:     {r['train_rows']} (post attention filter)")
    print(f"  Test rows:      {r['test_rows']} (post attention filter)")
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
    parser = argparse.ArgumentParser(description="Run Experiment A")
    parser.add_argument("--debug", action="store_true", help="Use small data subset")
    args = parser.parse_args()
    run_experiment_a(debug=args.debug)
