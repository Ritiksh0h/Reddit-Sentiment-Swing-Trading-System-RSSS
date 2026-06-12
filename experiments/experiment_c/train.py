#!/usr/bin/env python3
"""
Module: experiments/experiment_c/train.py
Purpose: Experiment C — Expanded Dataset + Combined Model.

Thesis: 3,210 training rows is insufficient. Adding r/stocks, r/investing,
r/options triples the dataset. More data may resolve the apparent
non-stationarity by providing stable signal estimates.

Dependency: Requires data/raw/merged_with_sentiment_expanded.parquet from Colab.
This file merges all 4 subreddits: wallstreetbets + stocks + investing + options.

Architecture (once expanded data is available):
  Step 1: Load expanded dataset
  Step 2: Apply density filter: post_count_1d >= 10 (now counts across ALL subreddits)
  Step 3: Rebuild features using pipeline/01_feature_builder.py --input-file flag
          Output: data/features/features_expanded.parquet
  Step 4: Train combined model (MARKET + REDDIT_SENTIMENT) on expanded features
  Step 5: Evaluate IC on 2024 test set
  Step 6: Run backtest and compare vs Experiment A

Expected outcome:
  - If C beats A: problem was data density (fixable with more subreddits)
  - If C ≈ A or worse: problem is structural non-stationarity (not fixable with data)

# WAITING FOR EXPANDED DATA
# Run Colab notebook to generate data/raw/merged_with_sentiment_expanded.parquet first.
# Experiments A and B can proceed independently.

Usage:
    python experiments/experiment_c/train.py
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
from utils.logger import get_logger

log = get_logger(__name__)

RESULTS_PATH = Path(__file__).parent / "results.json"
SPY_2024_RETURN = 0.2605
QQQ_2024_RETURN = 0.2550  # fallback; overridden by live fetch if yfinance available

try:
    import yfinance as _yf

    def _fetch_2024_return(ticker: str) -> float:
        data = _yf.download(ticker, start="2024-01-01", end="2024-12-31",
                            auto_adjust=True, progress=False)
        if data.empty:
            return {"SPY": SPY_2024_RETURN, "QQQ": QQQ_2024_RETURN}.get(ticker, 0.0)
        return float(data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)
except ImportError:
    def _fetch_2024_return(ticker: str) -> float:
        return {"SPY": SPY_2024_RETURN, "QQQ": QQQ_2024_RETURN}.get(ticker, 0.0)


def _build_expanded_features(input_parquet: str, output_parquet: str) -> None:
    """
    Build the feature matrix from the expanded parquet using the same pipeline
    logic as pipeline/01_feature_builder.py, but with configurable input/output
    paths. Imports internal functions directly — does NOT modify the pipeline script.
    """
    import importlib.util

    from config.settings import TRAIN_END, TEST_START

    # pipeline/01_feature_builder.py starts with a digit — can't use normal import syntax
    _spec = importlib.util.spec_from_file_location(
        "feature_builder",
        Path(__file__).parent.parent.parent / "pipeline" / "01_feature_builder.py",
    )
    _fb = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fb)

    load_focus_tickers = _fb.load_focus_tickers
    get_trading_days = _fb.get_trading_days
    compute_reddit_features = _fb.compute_reddit_features
    load_all_market_features = _fb.load_all_market_features
    apply_quality_filters = _fb.apply_quality_filters
    validate_no_leakage = _fb.validate_no_leakage
    FINAL_COL_ORDER = _fb.FINAL_COL_ORDER

    log.info("loading_expanded_parquet", path=input_parquet)
    reddit_raw = pd.read_parquet(input_parquet)
    log.info("expanded_parquet_loaded", rows=len(reddit_raw))

    # Deduplicate on (post_id, ticker) — keep highest-confidence match
    reddit_raw = (
        reddit_raw
        .sort_values("confidence", ascending=False)
        .drop_duplicates(subset=["post_id", "ticker"])
        .reset_index(drop=True)
    )

    # Filter to focus tickers
    tickers = load_focus_tickers()
    reddit_raw = reddit_raw[reddit_raw["ticker"].isin(tickers)].reset_index(drop=True)
    log.info("focus_filtered", tickers=len(tickers), rows=len(reddit_raw))

    if reddit_raw.empty:
        raise RuntimeError("No rows remain after filtering to focus tickers.")

    # Date range from data
    ts_min = pd.to_datetime(reddit_raw["timestamp"], utc=True).min()
    ts_max = pd.to_datetime(reddit_raw["timestamp"], utc=True).max()
    data_start = ts_min.date().isoformat()
    data_end = ts_max.date().isoformat()

    trading_days = get_trading_days(data_start, data_end)
    log.info("trading_days", count=len(trading_days))

    # Reddit features
    reddit_features = compute_reddit_features(reddit_raw, tickers, trading_days)
    reddit_features["date"] = reddit_features["date"].astype(str)
    log.info("reddit_features_done", rows=len(reddit_features))

    # Market features
    market_features = load_all_market_features(
        tickers, start_date=data_start, end_date=data_end, force_refresh=False
    )

    # Join
    combined = pd.merge(reddit_features, market_features, on=["ticker", "date"], how="inner")
    log.info("features_joined", rows=len(combined))

    # Quality filters + split + leakage check
    combined = apply_quality_filters(combined)
    combined["split"] = np.where(combined["date"] < TEST_START, "train", "test")
    available_cols = [c for c in FINAL_COL_ORDER if c in combined.columns]
    extra_cols = [c for c in combined.columns if c not in FINAL_COL_ORDER]
    combined = combined[available_cols + extra_cols]
    validate_no_leakage(combined)

    # Save
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_parquet, index=False)
    log.info(
        "expanded_features_saved",
        path=output_parquet,
        rows=len(combined),
        train=int((combined["split"] == "train").sum()),
        test=int((combined["split"] == "test").sum()),
    )


def run_experiment_c() -> dict:
    """Train, evaluate, and backtest Experiment C. Returns results dict."""
    # ------------------------------------------------------------------
    # HARD GATE: expanded data must exist before proceeding
    # ------------------------------------------------------------------
    if not os.path.exists(EXPANDED_PARQUET_PATH):
        print()
        print("=" * 60)
        print("EXPERIMENT C: WAITING FOR EXPANDED DATA")
        print("=" * 60)
        print(f"  Required: {EXPANDED_PARQUET_PATH}")
        print("  Status:   NOT FOUND")
        print()
        print("  Action required:")
        print("  1. Run the Colab notebook to scrape and merge all 4 subreddits:")
        print("     wallstreetbets + stocks + investing + options")
        print("  2. Drop merged_with_sentiment_expanded.parquet into data/raw/")
        print("  3. Re-run this script")
        print()
        print("  Experiments A and B can run independently while waiting.")
        print("=" * 60)
        print()
        sys.exit(0)

    spy_return = _fetch_2024_return("SPY")
    qqq_return = _fetch_2024_return("QQQ")
    log.info("benchmarks_fetched", spy=round(spy_return, 4), qqq=round(qqq_return, 4))
    log.info("experiment_c_start", expanded_path=EXPANDED_PARQUET_PATH)

    from experiments.shared.backtest import run_backtest
    from experiments.shared.trainer import train_xgboost, evaluate_ic
    from pipeline.feature_schema import MARKET_FEATURES, TARGET_COL

    # ------------------------------------------------------------------
    # 1. Check for pre-built expanded features, or rebuild them
    #    The feature builder CLI has no --input-file/--output-file flags,
    #    so we import its internal functions and run the pipeline directly.
    # ------------------------------------------------------------------
    if not os.path.exists(EXPANDED_FEATURES_PATH):
        log.info("building_expanded_features", input=EXPANDED_PARQUET_PATH)
        _build_expanded_features(EXPANDED_PARQUET_PATH, EXPANDED_FEATURES_PATH)
        log.info("expanded_features_built", path=EXPANDED_FEATURES_PATH)
    else:
        log.info("expanded_features_found", path=EXPANDED_FEATURES_PATH)

    # ------------------------------------------------------------------
    # 2. Load expanded features
    # ------------------------------------------------------------------
    df = pd.read_parquet(EXPANDED_FEATURES_PATH)
    log.info("features_loaded", rows=len(df))

    # ------------------------------------------------------------------
    # 3. Apply density filter (post_count_1d >= 10 across all subreddits)
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
    # 4. Train / test split
    # ------------------------------------------------------------------
    COMBINED_FEATURES = MARKET_FEATURES + [
        "avg_sentiment_1d",
        "avg_sentiment_3d",
        "avg_sentiment_hc",
        "weighted_sentiment",
        "bullish_ratio",
        "sentiment_accel",
        "sentiment_std",
    ]

    train = df_filtered[df_filtered["split"] == "train"].copy()
    test = df_filtered[df_filtered["split"] == "test"].copy()

    log.info("split_sizes", train_rows=len(train), test_rows=len(test))

    if len(train) < 50:
        log.error("insufficient_train_rows", rows=len(train))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Train combined model
    # ------------------------------------------------------------------
    log.info("training_combined_model", features=len(COMBINED_FEATURES))
    model = train_xgboost(train, COMBINED_FEATURES, TARGET_COL)
    log.info("model_trained")

    # ------------------------------------------------------------------
    # 6. Evaluate IC
    # ------------------------------------------------------------------
    ic_test = evaluate_ic(model, test, COMBINED_FEATURES, TARGET_COL)
    log.info("ic_evaluated", ic_test=round(ic_test, 4))

    # ------------------------------------------------------------------
    # 7. Backtest
    # ------------------------------------------------------------------
    log.info("backtest_start")
    backtest_results = run_backtest(
        model=model,
        test_df=test,
        feature_cols=COMBINED_FEATURES,
        pre_filter_fn=None,
        spy_return=spy_return,
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
    total_return = backtest_results.get("total_return", 0)
    results = {
        "experiment": "C",
        "thesis": "expanded_dataset_combined_model",
        "expanded_data_path": EXPANDED_PARQUET_PATH,
        "features": COMBINED_FEATURES,
        "density_filter_min_posts": ATTENTION_FILTER_MIN_POSTS,
        "total_rows_before_filter": len(df),
        "pct_rows_retained": round(pct_retained, 4),
        "train_rows": len(train),
        "test_rows": len(test),
        "ic_test": round(ic_test, 6),
        "passes_ic_threshold": bool(ic_test > 0.05),
        "passes_sharpe_threshold": bool(backtest_results["sharpe_ratio"] > 1.0),
        "qqq_return": round(qqq_return, 6),
        "beats_qqq": bool(total_return > qqq_return),
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


def _print_summary(r: dict) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT C RESULTS")
    print("=" * 60)
    print(f"  Thesis:         {r['thesis']}")
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
    print(f"  QQQ return:     {r.get('qqq_return', 0)*100:.1f}%")
    print(f"  Beats QQQ:      {r.get('beats_qqq', False)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_experiment_c()
