#!/usr/bin/env python3
"""
Module: pipeline/03_train_models.py
Purpose: Train 3 XGBoost models (Market Only, Reddit Only, Combined).
         Compare IC scores. Gate: Combined IC > Market IC + 0.005 or stop.

Phase: 1 — Research Pipeline
Input:  data/features/features.parquet
Output: models/registry/{market_only,reddit_only,combined}.json
        reports/model_comparison.json

Usage:
    python pipeline/03_train_models.py
    python pipeline/03_train_models.py --debug
Last modified: 2026-06-11
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from experiments.shared.metrics import compute_ic

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import FEATURES_PARQUET, MODELS_DIR, REPORTS_DIR
from config.thresholds import (
    MIN_IC_THRESHOLD,
    REDDIT_ADDS_VALUE_IC,
    N_PERMUTATIONS,
    PVALUE_THRESHOLD,
)
from pipeline.feature_schema import (
    MARKET_FEATURES,
    REDDIT_FEATURES,
    TARGET_COL,
)
from utils.logger import get_logger

log = get_logger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed. Run: pip install joblib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# IC utilities
# ---------------------------------------------------------------------------

def compute_ic_series(df: pd.DataFrame, preds: np.ndarray) -> dict:
    """Compute IC on full test set and optionally by year/quarter."""
    df = df.copy()
    df["pred"] = preds
    df["date_dt"] = pd.to_datetime(df["date"])

    # Full IC
    ic_full = compute_ic(df["target_return_5d"].values, df["pred"].values)

    # Monthly IC
    df["year_month"] = df["date_dt"].dt.to_period("M")
    monthly_ics = (
        df.groupby("year_month")
        .apply(lambda g: compute_ic(g["target_return_5d"].values, g["pred"].values))
        .dropna()
    )

    return {
        "ic_full": round(ic_full, 6),
        "ic_monthly_mean": round(float(monthly_ics.mean()), 6) if len(monthly_ics) > 0 else 0.0,
        "ic_monthly_std": round(float(monthly_ics.std()), 6) if len(monthly_ics) > 0 else 0.0,
        "ic_monthly_values": {str(k): round(v, 4) for k, v in monthly_ics.items()},
    }


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame, feature_group: str) -> tuple[list[str], pd.DataFrame]:
    """Return feature names and array for given group."""
    if feature_group == "market":
        cols = [c for c in MARKET_FEATURES if c in df.columns]
    elif feature_group == "reddit":
        cols = [c for c in REDDIT_FEATURES if c in df.columns]
    elif feature_group == "combined":
        cols = [c for c in MARKET_FEATURES + REDDIT_FEATURES if c in df.columns]
    else:
        raise ValueError(f"Unknown feature_group: {feature_group}")

    expected = MARKET_FEATURES if feature_group == "market" else (
        REDDIT_FEATURES if feature_group == "reddit" else MARKET_FEATURES + REDDIT_FEATURES
    )
    missing = [c for c in expected if c not in df.columns]
    if missing:
        log.warning("features_missing_from_df", group=feature_group, missing=missing)

    return cols, df[cols].copy()


# ---------------------------------------------------------------------------
# XGBoost model (shared hyperparameters per spec)
# ---------------------------------------------------------------------------

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
    "verbosity": 0,
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    early_stopping_rounds: int = 30,
    eval_fraction: float = 0.15,
) -> xgb.XGBRegressor:
    """Train XGBoost with early stopping on a time-ordered validation slice."""
    # Time-ordered validation split — use the last eval_fraction of train data
    n = len(X_train)
    val_idx = int(n * (1 - eval_fraction))
    X_tr, X_val = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
    y_tr, y_val = y_train.iloc[:val_idx], y_train.iloc[val_idx:]

    params = XGBOOST_PARAMS.copy()
    params["early_stopping_rounds"] = early_stopping_rounds

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration if hasattr(model, "best_iteration") else params["n_estimators"]
    log.info("model_trained", name=model_name, best_iter=best_iter)
    return model


def save_model(model: xgb.XGBRegressor, feature_cols: list[str], model_name: str,
               ic_metrics: dict) -> Path:
    """Save model + metadata to model registry."""
    save_dir = MODELS_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = save_dir / "model.json"
    model.save_model(str(model_path))

    # Save joblib for predict convenience
    joblib.dump(model, save_dir / "model.pkl")

    # Save metadata
    meta = {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "xgboost_params": XGBOOST_PARAMS,
        "ic_metrics": ic_metrics,
        "saved_at": pd.Timestamp.now().isoformat(),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("model_saved", path=str(save_dir), ic=ic_metrics.get("ic_full"))
    return save_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Phase 1 — Train XGBoost Models")
    parser.add_argument("--debug", action="store_true",
                        help="Use 2 tickers, fast convergence")
    args = parser.parse_args()

    if not FEATURES_PARQUET.exists():
        log.error("features_not_found", path=str(FEATURES_PARQUET))
        print(f"ERROR: {FEATURES_PARQUET} not found. Run 01_feature_builder.py first.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PARQUET)
    log.info("features_loaded", rows=len(df), cols=df.columns.tolist())

    # Enforce time-based split — NEVER shuffle
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        log.error("split_error", train_rows=len(train_df), test_rows=len(test_df))
        print("ERROR: train or test split is empty.")
        sys.exit(1)

    if args.debug:
        tickers = train_df["ticker"].unique()[:2]
        train_df = train_df[train_df["ticker"].isin(tickers)]
        test_df = test_df[test_df["ticker"].isin(tickers)]
        # Speed up training
        XGBOOST_PARAMS["n_estimators"] = 50
        log.info("debug_mode", tickers=list(tickers))

    log.info(
        "data_summary",
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_dates=f"{train_df['date'].min()} → {train_df['date'].max()}",
        test_dates=f"{test_df['date'].min()} → {test_df['date'].max()}",
    )

    # Drop rows with NaN target
    train_df = train_df.dropna(subset=[TARGET_COL])
    test_df = test_df.dropna(subset=[TARGET_COL])
    log.info("after_target_dropna", train_rows=len(train_df), test_rows=len(test_df))

    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    results = {}

    for group in ("market", "reddit", "combined"):
        log.info("training_group", group=group)
        feat_cols, X_train_g = select_features(train_df, group)
        _, X_test_g = select_features(test_df, group)

        # Fill NaN with column median from train (no leakage: only train stats)
        # Reddit features: NaN means zero activity — fill with 0 for reddit, median for market
        if group == "reddit":
            X_train_g = X_train_g.fillna(0)
            X_test_g = X_test_g.fillna(0)
        else:
            train_medians = X_train_g.median()
            X_train_g = X_train_g.fillna(train_medians)
            X_test_g = X_test_g.fillna(train_medians)

        model = train_model(X_train_g, y_train, model_name=group)
        preds_test = model.predict(X_test_g)

        ic_metrics = compute_ic_series(test_df, preds_test)
        ic_metrics["group"] = group
        ic_metrics["n_train"] = len(train_df)
        ic_metrics["n_test"] = len(test_df)
        ic_metrics["n_features"] = len(feat_cols)

        save_model(model, feat_cols, group, ic_metrics)
        results[group] = ic_metrics

        log.info(
            "model_ic_result",
            group=group,
            ic_full=ic_metrics["ic_full"],
            ic_monthly_mean=ic_metrics["ic_monthly_mean"],
        )

    # --- Decision Gate ---
    ic_market = results["market"]["ic_full"]
    ic_combined = results["combined"]["ic_full"]
    ic_diff = ic_combined - ic_market

    verdict = "REDDIT_ADDS_VALUE" if ic_diff > REDDIT_ADDS_VALUE_IC else "REDDIT_NOT_ADDITIVE"

    if verdict == "REDDIT_NOT_ADDITIVE":
        log.warning(
            "reddit_not_additive",
            ic_market=ic_market,
            ic_combined=ic_combined,
            diff=round(ic_diff, 4),
            threshold=REDDIT_ADDS_VALUE_IC,
        )

    # Check minimum IC threshold
    if ic_combined < MIN_IC_THRESHOLD:
        log.warning(
            "ic_below_threshold",
            ic_combined=ic_combined,
            threshold=MIN_IC_THRESHOLD,
        )

    # Save comparison report
    comparison = {
        "run_date": pd.Timestamp.now().isoformat(),
        "models": results,
        "comparison": {
            "ic_market": ic_market,
            "ic_reddit": results["reddit"]["ic_full"],
            "ic_combined": ic_combined,
            "ic_diff_combined_vs_market": round(ic_diff, 6),
            "reddit_adds_value_threshold": REDDIT_ADDS_VALUE_IC,
            "verdict": verdict,
            "combined_above_min_threshold": ic_combined >= MIN_IC_THRESHOLD,
        },
    }

    report_path = REPORTS_DIR / "model_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info("model_comparison_saved", path=str(report_path))

    # Console summary
    print("\n=== MODEL IC COMPARISON ===")
    print(f"  Market Only:    IC = {ic_market:+.4f}")
    print(f"  Reddit Only:    IC = {results['reddit']['ic_full']:+.4f}")
    print(f"  Combined:       IC = {ic_combined:+.4f}  (diff vs market: {ic_diff:+.4f})")
    print()
    print(f"  Threshold (min Reddit improvement): {REDDIT_ADDS_VALUE_IC}")
    print(f"  VERDICT: {verdict}")
    if verdict == "REDDIT_NOT_ADDITIVE":
        print("\n  ⚠️  GATE: Combined IC not meaningfully better than market-only.")
        print("           Per spec: stop here and reassess Reddit features.")
    elif ic_combined < MIN_IC_THRESHOLD:
        print(f"\n  ⚠️  WARNING: Combined IC {ic_combined:.4f} < min threshold {MIN_IC_THRESHOLD}")
    else:
        print("\n  ✅ Continue to Script 04 (backtests)")
    print("=" * 40)


if __name__ == "__main__":
    main()
