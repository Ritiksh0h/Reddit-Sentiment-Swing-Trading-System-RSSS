#!/usr/bin/env python3
"""
Module: experiments/walk_forward.py
Purpose: Walk-forward validation for Experiment C architecture.
         Three rolling windows confirm whether IC is consistent across years
         or regime-specific.

Decision gates:
  Min IC >= 0.05 → ROBUST
  Min IC >= 0.03 → ACCEPTABLE
  Min IC >= 0.00 → REGIME-DEPENDENT (do not proceed to Phase 3)
  Min IC <  0.00 → FRAGILE (stop, reassess architecture)

Usage:
    python experiments/walk_forward.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

FEATURES = [
    "returns_1d", "returns_5d", "returns_20d",
    "rsi_14", "atr_14", "relative_volume",
    "dist_from_20ma", "dist_from_50ma",
    "avg_sentiment_1d", "avg_sentiment_3d", "weighted_sentiment",
    "sentiment_std", "sentiment_accel", "bullish_ratio",
    "post_count_1d", "mention_growth_1d", "mention_growth_7d",
]

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    objective="reg:squarederror",
)

WINDOWS = [
    ("2019-2021", [2019, 2020, 2021], "2022", [2022]),
    ("2020-2022", [2020, 2021, 2022], "2023", [2023]),
    ("2021-2023", [2021, 2022, 2023], "2024", [2024]),
]


def run_walk_forward(features_path: str = "data/features/features_expanded.parquet") -> dict:
    df = pd.read_parquet(features_path)
    df = df[df["post_count_1d"] >= 10].copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    active_features = [f for f in FEATURES if f in df.columns]
    if len(active_features) < len(FEATURES):
        missing = set(FEATURES) - set(active_features)
        print(f"Warning: {len(missing)} features missing from store: {missing}")

    print()
    print("=" * 65)
    print("WALK-FORWARD VALIDATION — Experiment C Architecture")
    print("=" * 65)
    print(f"{'Train':<15} {'Test':<8} {'IC':>8} {'n_train':>8} {'n_test':>8}  Robust?")
    print("-" * 65)

    results = []
    for train_label, train_years, test_label, test_years in WINDOWS:
        train = df[df["year"].isin(train_years)]
        test = df[df["year"].isin(test_years)]

        if len(train) < 100 or len(test) < 30:
            print(f"{train_label:<15} {test_label:<8}  insufficient data")
            continue

        X_tr = train[active_features].fillna(0)
        y_tr = train["target_return_5d"]
        X_te = test[active_features].fillna(0)
        y_te = test["target_return_5d"]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        ic, pval = stats.spearmanr(pred, y_te)

        robust = "YES" if ic >= 0.05 else "MARGINAL" if ic >= 0.03 else "NO"
        results.append({
            "window": f"{train_label}→{test_label}",
            "train_label": train_label,
            "test_label": test_label,
            "ic": round(float(ic), 6),
            "pval": round(float(pval), 6),
            "n_train": len(train),
            "n_test": len(test),
            "robust": robust,
        })
        print(
            f"{train_label:<15} {test_label:<8} {ic:>8.4f} "
            f"{len(train):>8} {len(test):>8}  {robust}"
        )

    if not results:
        print("ERROR: No windows had sufficient data.")
        sys.exit(1)

    ics = [r["ic"] for r in results]
    mean_ic = float(np.mean(ics))
    min_ic = float(np.min(ics))
    std_ic = float(np.std(ics))

    print()
    print(f"Mean IC across windows:   {mean_ic:.4f}")
    print(f"Min IC across windows:    {min_ic:.4f}")
    print(f"Std IC across windows:    {std_ic:.4f}")
    print()

    if min_ic >= 0.05:
        verdict = "ROBUST — signal consistent across all periods"
        phase3_decision = "PROCEED"
    elif min_ic >= 0.03:
        verdict = "ACCEPTABLE — positive signal in all periods, some variance"
        phase3_decision = "PROCEED with caution, note regime risk in production"
    elif min_ic >= 0.0:
        verdict = "REGIME-DEPENDENT — signal positive but inconsistent"
        phase3_decision = "HOLD — add regime detection before portfolio engine"
    else:
        verdict = "FRAGILE — signal collapses in at least one period"
        phase3_decision = "STOP — reassess architecture"

    print(f"VERDICT:  {verdict}")
    print(f"DECISION: {phase3_decision}")
    print()

    output = {
        "walk_forward_results": results,
        "mean_ic": round(mean_ic, 6),
        "min_ic": round(min_ic, 6),
        "std_ic": round(std_ic, 6),
        "verdict": verdict,
        "phase3_decision": phase3_decision,
        "robust": min_ic >= 0.03,
    }

    out_path = Path("experiments/walk_forward_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")

    return output


if __name__ == "__main__":
    run_walk_forward()
