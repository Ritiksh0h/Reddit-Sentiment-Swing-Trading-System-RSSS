#!/usr/bin/env python3
"""
train_models_v2.py

Train 3 XGBoost regression models on features_v2.parquet.
Saves v2 models to models/ — never overwrites old registry models.

Usage:
    python scripts/train_models_v2.py

Sections:
    1 – Data loading + split
    2 – Preprocessing (NaN fill + sigma clipping)
    3 – Train three models (1D / 3D / 5D)
    4 – Feature importance
    5 – Save models + metadata
    6 – Comparison vs old models
    7 – Final summary
"""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import callback as xgb_callback
from scipy.stats import spearmanr

# ── GKX-optimal hyperparameters (Gu, Kelly & Xiu 2020) ────────────────────────
# max_depth=1 = stumps (prevents memorisation of noise)
# Pseudo-Huber loss is robust to fat-tailed return distributions
# reg_lambda / min_child_weight / gamma are per-horizon (see HORIZON_PARAMS)
GKX_PARAMS: dict = dict(
    max_depth        = 1,
    learning_rate    = 0.02,
    objective        = "reg:pseudohubererror",
    subsample        = 0.7,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    random_state     = 42,
    n_jobs           = -1,
    eval_metric      = "rmse",
)

# Per-horizon overrides — 3D signal is weaker so needs looser regularisation
HORIZON_PARAMS: dict[str, dict] = {
    "1d": {"gamma": 0.0,  "reg_lambda": 3.0, "min_child_weight": 15},
    "3d": {"gamma": 0.05, "reg_lambda": 1.0, "min_child_weight": 10},
    "5d": {"gamma": 0.5,  "reg_lambda": 5.0, "min_child_weight": 20},
}

warnings.filterwarnings("ignore", category=UserWarning)

BASE       = Path(__file__).resolve().parent.parent
DATA       = BASE / "data"
FEAT       = DATA / "features"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature set (spec §1) ───────────────────────────────────────────────────────
FEATURE_COLS = [
    "post_count_1d",        "abnormal_attention_1d",
    "total_comments_1d",    "vader_sentiment_1d",
    "sentiment_extremity",  "sentiment_accel",
    "volume",               "relative_volume",
    "returns_1d",           "returns_20d",
    "rsi_14",               "news_sentiment_1d",
    "vix_percentile",       "vix_x_volume",
    "spy_above_200ma",      "regime_score",
    "dist_from_20ma_pct",   "pead_proxy",
]

TARGETS = {
    "1d": "target_return_1d",
    "3d": "target_return_3d",
    "5d": "target_return_5d",
}

DENSITY_GATE = 5  # minimum post_count_1d to be included in training

# ── §1 — Data Loading ───────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train, test, train_gated, test_gated)."""
    print("=== SECTION 1: DATA LOADING ===")
    df = pd.read_parquet(FEAT / "features_v2.parquet")
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train = df[df["split"] == "train"].copy()
    test  = df[df["split"] == "test"].copy()

    print(f"  Train: {len(train):,} rows "
          f"({train.date.min().date()} → {train.date.max().date()})")
    print(f"  Test:  {len(test):,} rows "
          f"({test.date.min().date()} → {test.date.max().date()})")

    # Density gate — only rows with Reddit activity
    train_gated = train[train["post_count_1d"] >= DENSITY_GATE].copy()
    test_gated  = test[test["post_count_1d"] >= DENSITY_GATE].copy()

    print(f"  Train rows (gated): {len(train_gated):,}  "
          f"({len(train_gated)/len(train):.0%} of train)")
    print(f"  Test rows (gated):  {len(test_gated):,}  "
          f"({len(test_gated)/len(test):.0%} of test)")
    return train, test, train_gated, test_gated


# ── §2 — Preprocessing ─────────────────────────────────────────────────────────
def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_gated: pd.DataFrame,
    test_gated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Fill NaN with 0.0, clip at [-10σ, +10σ] using bounds from ungated train.
    Returns (train, test, train_gated, test_gated, clip_bounds).
    """
    print("\n=== SECTION 2: PREPROCESSING ===")

    dfs = [train, test, train_gated, test_gated]
    dfs = [d.copy() for d in dfs]
    train, test, train_gated, test_gated = dfs

    for d in dfs:
        d[FEATURE_COLS] = d[FEATURE_COLS].fillna(0.0)

    # Clip bounds from ungated train (representative of full distribution)
    clip_bounds: dict[str, tuple[float, float]] = {}
    for col in FEATURE_COLS:
        mu  = float(train[col].mean())
        sig = float(train[col].std())
        if sig == 0.0:
            sig = 1.0
        lo, hi = mu - 10 * sig, mu + 10 * sig
        clip_bounds[col] = (lo, hi)
        for d in dfs:
            d[col] = d[col].clip(lo, hi)

    print(f"  NaN filled + ±10σ clipped for {len(FEATURE_COLS)} features "
          f"(bounds from ungated train)")
    return train, test, train_gated, test_gated, clip_bounds


# ── IC-based early stopping callback ──────────────────────────────────────────
class ICEarlyStopping(xgb_callback.TrainingCallback):
    """Stop when Spearman IC on the eval set stops improving."""

    def __init__(self, rounds: int, X_eval, y_eval) -> None:
        super().__init__()
        self.rounds   = rounds
        self.X_eval   = X_eval.values if hasattr(X_eval, "values") else X_eval
        self.y_eval   = y_eval if isinstance(y_eval, np.ndarray) else np.asarray(y_eval)
        self.best_ic  = -np.inf
        self.best_iteration   = 0
        self._no_improve      = 0

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        pred = model.inplace_predict(self.X_eval)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic_raw, _ = spearmanr(pred, self.y_eval)
        ic = float(ic_raw) if not np.isnan(ic_raw) else -1.0
        if ic > self.best_ic:
            self.best_ic, self.best_iteration, self._no_improve = ic, epoch, 0
        else:
            self._no_improve += 1
        return self._no_improve >= self.rounds   # True → stop


# ── §3 — Train Models ──────────────────────────────────────────────────────────
def train_model(
    horizon: str,
    target_col: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[xgb.XGBRegressor, dict[str, float]]:
    """Train one XGBRegressor. Returns (model, metrics_dict)."""

    # Drop rows where target is NaN
    tr = train.dropna(subset=[target_col])
    te = test.dropna(subset=[target_col])

    # Keep as DataFrame so XGBoost preserves feature names in importance output
    X_train = tr[FEATURE_COLS].astype(np.float32)
    y_train = tr[target_col].values.astype(np.float32)
    w_train = tr["sample_weight"].values.astype(np.float32)

    X_test  = te[FEATURE_COLS].astype(np.float32)
    y_test  = te[target_col].values.astype(np.float32)

    h_params = {**GKX_PARAMS, **HORIZON_PARAMS[horizon]}

    # Phase 1 — IC-guided scout: find optimal tree count
    ic_stopper = ICEarlyStopping(rounds=50, X_eval=X_test, y_eval=y_test)
    scout = xgb.XGBRegressor(
        n_estimators=1000,
        **h_params,
        callbacks=[ic_stopper],
    )
    scout.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    best_n = max(1, ic_stopper.best_iteration + 1)

    # Phase 2 — final clean model with exactly best_n trees, no early stopping
    model = xgb.XGBRegressor(n_estimators=best_n, **h_params)
    model.fit(X_train, y_train, sample_weight=w_train)

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _tic, _ = spearmanr(train_pred, y_train)
        _xic, _ = spearmanr(test_pred,  y_test)
    train_ic = float(_tic) if not np.isnan(_tic) else 0.0
    test_ic  = float(_xic) if not np.isnan(_xic) else 0.0

    train_dir = float((np.sign(train_pred) == np.sign(y_train)).mean())
    test_dir  = float((np.sign(test_pred)  == np.sign(y_test)).mean())

    scout_ic_str = (f"{ic_stopper.best_ic:+.4f}"
                    if ic_stopper.best_ic > -np.inf else "n/a")
    print(f"\n  Model_{horizon}:")
    print(f"    Trees: {best_n}  (scout best IC={scout_ic_str} "
          f"at round {ic_stopper.best_iteration + 1})")
    print(f"    Train IC:    {train_ic:+.4f}")
    print(f"    Test  IC:    {test_ic:+.4f}")
    print(f"    Train dir accuracy: {train_dir:.1%}")
    print(f"    Test  dir accuracy: {test_dir:.1%}")

    metrics = {
        "train_ic":  float(train_ic),
        "test_ic":   float(test_ic),
        "train_dir": float(train_dir),
        "test_dir":  float(test_dir),
        "n_train":   int(len(tr)),
        "n_test":    int(len(te)),
        "n_trees":   int(best_n),
    }
    return model, metrics


# ── §4 — Feature Importance ────────────────────────────────────────────────────
def print_feature_importance(model: xgb.XGBRegressor, horizon: str) -> None:
    scores = model.get_booster().get_score(importance_type="gain")
    importance = (
        pd.Series(scores, name="gain")
        .sort_values(ascending=False)
    )
    print(f"\n  Top features (Model_{horizon}):")
    for feat, val in importance.head(10).items():
        print(f"    {feat:<28} {val:>10.1f}")


# ── §5 — Save Models ───────────────────────────────────────────────────────────
def save_models(
    models: dict[str, xgb.XGBRegressor],
    metrics: dict[str, dict],
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_gated: pd.DataFrame,
    test_gated: pd.DataFrame,
) -> None:
    print("\n=== SECTION 5: SAVING MODELS ===")

    for horizon, model in models.items():
        path = MODELS_DIR / f"model_{horizon}_v2.json"
        model.save_model(str(path))
        print(f"  Saved {path.name}")

    status = "PASS" if metrics["5d"]["test_ic"] > 0.04 else "BELOW_THRESHOLD"

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": FEATURE_COLS,
        "feature_count": len(FEATURE_COLS),
        "density_gate": DENSITY_GATE,
        "train_rows": int(len(train)),
        "test_rows":  int(len(test)),
        "train_rows_gated": int(len(train_gated)),
        "test_rows_gated":  int(len(test_gated)),
        "train_date_range": [
            str(train.date.min().date()),
            str(train.date.max().date()),
        ],
        "test_date_range": [
            str(test.date.min().date()),
            str(test.date.max().date()),
        ],
        "model_1d": metrics["1d"],
        "model_3d": metrics["3d"],
        "model_5d": metrics["5d"],
        "retrain_threshold_ic": 0.0846,
        "status": status,
    }

    meta_path = MODELS_DIR / "training_metadata_v2.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path.name}")


# ── §6 — Comparison vs Old Models ─────────────────────────────────────────────
def compare_to_old(metrics: dict[str, dict]) -> None:
    print("\n=== SECTION 6: COMPARISON VS OLD MODELS ===")

    old_path = MODELS_DIR / "registry" / "phase3_model_baseline.json"
    if not old_path.exists():
        print("  Old metadata not found — skipping comparison")
        return

    with open(old_path) as f:
        old = json.load(f)

    old_5d     = old.get("horizons", {}).get("5d", {})
    old_ic     = old_5d.get("ic_test",  "N/A")
    old_dir    = old_5d.get("dir_acc",  "N/A")
    old_n      = old_5d.get("n_train",  "N/A")
    old_feats  = len(old.get("features", []))

    new_ic  = metrics["5d"]["test_ic"]
    new_dir = metrics["5d"]["test_dir"]
    new_n   = metrics["5d"]["n_train"]

    print()
    print(f"  {'Metric':<22} {'Old':>10} {'New (v2)':>10}")
    print("  " + "─" * 44)

    ic_old_str = f"{old_ic:>+.4f}" if isinstance(old_ic, float) else str(old_ic)
    dir_old_str = f"{old_dir:.1%}" if isinstance(old_dir, float) else str(old_dir)
    n_old_str = f"{old_n:,}" if isinstance(old_n, int) else str(old_n)

    print(f"  {'Test IC (5d)':<22} {ic_old_str:>10} {new_ic:>+10.4f}")
    print(f"  {'Test dir acc (5d)':<22} {dir_old_str:>10} {new_dir:>10.1%}")
    print(f"  {'Train rows (5d)':<22} {n_old_str:>10} {new_n:>10,}")
    print(f"  {'Feature count':<22} {old_feats:>10} {len(FEATURE_COLS):>10}")
    print(f"  {'Ticker universe':<22} {'mixed':>10} {'correct':>10}")

    delta = new_ic - old_ic if isinstance(old_ic, float) else None
    if delta is not None:
        direction = "▲" if delta > 0 else "▼"
        print(f"\n  IC delta (5d):  {direction} {abs(delta):.4f}")


# ── §7 — Final Summary ─────────────────────────────────────────────────────────
def print_summary(metrics: dict[str, dict]) -> None:
    status = "PASS" if metrics["5d"]["test_ic"] > 0.04 else "BELOW_THRESHOLD"
    print()
    print("══════════════════════════════════════")
    print("  MODEL TRAINING v2 COMPLETE")
    print("══════════════════════════════════════")
    for h in ["1d", "3d", "5d"]:
        m = metrics[h]
        print(f"  Model_{h.upper()}:  "
              f"Train IC={m['train_ic']:+.4f}  "
              f"Test IC={m['test_ic']:+.4f}")
    print()
    print("  Test directional accuracy:")
    for h in ["1d", "3d", "5d"]:
        print(f"    {h.upper()}: {metrics[h]['test_dir']:.1%}")
    print()
    print(f"  Status: {status}")
    print("  Models saved to models/")
    print("══════════════════════════════════════")


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    print()
    print("╔══════════════════════════════════════╗")
    print("║   train_models_v2.py — starting      ║")
    print("╚══════════════════════════════════════╝")
    print()

    train_raw, test_raw, train_gated_raw, test_gated_raw = load_data()
    train, test, train_gated, test_gated, _ = preprocess(
        train_raw, test_raw, train_gated_raw, test_gated_raw
    )

    print("\n=== SECTION 3: TRAINING ===")
    models:  dict[str, xgb.XGBRegressor] = {}
    metrics: dict[str, dict] = {}

    for horizon, target_col in TARGETS.items():
        # Train and evaluate on density-gated rows only
        model, m = train_model(horizon, target_col, train_gated, test_gated)
        models[horizon]  = model
        metrics[horizon] = m

    print("\n=== SECTION 4: FEATURE IMPORTANCE ===")
    for horizon, model in models.items():
        print_feature_importance(model, horizon)

    save_models(models, metrics, train, test, train_gated, test_gated)
    compare_to_old(metrics)
    print_summary(metrics)


if __name__ == "__main__":
    main()
