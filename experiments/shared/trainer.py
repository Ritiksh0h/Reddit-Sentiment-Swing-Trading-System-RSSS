"""
Module: experiments/shared/trainer.py
Purpose: Shared XGBoost training and IC evaluation. Used by all three experiments.
Phase: 2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

XGB_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


def train_xgboost(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict = XGB_PARAMS,
) -> xgb.XGBRegressor:
    """Train XGBoost regressor. Returns fitted model."""
    X = train_df[feature_cols].fillna(0)
    y = train_df[target_col]
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def evaluate_ic(
    model: xgb.XGBRegressor,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> float:
    """Return Spearman IC on test set."""
    X = test_df[feature_cols].fillna(0)
    y = test_df[target_col]
    pred = model.predict(X)
    ic, _ = spearmanr(pred, y)
    return float(ic) if np.isfinite(ic) else 0.0


def predict(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Generate predictions for a dataframe."""
    X = df[feature_cols].fillna(0)
    return model.predict(X)
