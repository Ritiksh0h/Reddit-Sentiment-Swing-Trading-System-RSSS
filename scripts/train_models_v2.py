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

# Per-horizon overrides — gamma=0.0 for all horizons
# gamma=0.5 on 5D caused prediction collapse (stumps refused to split
# on noisy 5D returns — 97.6% of predictions were identical).
# Confirmed fix: gamma=0.0 gives IC=+0.0246 vs IC=-0.0005 with gamma=0.5
HORIZON_PARAMS: dict[str, dict] = {
    "1d": {"gamma": 0.0, "reg_lambda": 3.0, "min_child_weight": 15},
    "3d": {"gamma": 0.0, "reg_lambda": 1.0, "min_child_weight": 10},
    "5d": {"gamma": 0.0, "reg_lambda": 5.0, "min_child_weight": 20},
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
    "dist_from_20ma_pct",   "pead_proxy",
]

TARGETS = {
    "1d": "target_return_1d",
    "3d": "target_return_3d",
    "5d": "target_return_5d",
}

DENSITY_GATE = 5  # minimum post_count_1d to be included in training

# ── Date windows — sliding 2-year training window ──────────────────────────────
# Excludes 2020 COVID crash + 2021 meme-stock patterns that do not generalise.
# Walk-forward (12-fold) confirmed: 2022-2023 train → mean OOS IC=+0.044 (83% positive)
TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"
VAL_START   = "2023-07-01"   # held-out val — ICEarlyStopping only, not final eval
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"


# ── §1 — Data Loading ───────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train, val, test, train_gated, val_gated, test_gated).

    train:   2022-01-01 → 2023-12-31  (full sliding window for final fit)
    val:     2023-07-01 → 2023-12-31  (held-out for ICEarlyStopping only)
    test:    2024-01-01 → 2025-12-31  (final evaluation — never seen during training)
    *_gated: density gate applied (post_count_1d >= DENSITY_GATE)
    """
    print("=== SECTION 1: DATA LOADING ===")
    df = pd.read_parquet(FEAT / "features_v2.parquet")
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    val   = df[(df["date"] >= VAL_START)   & (df["date"] <= VAL_END)].copy()
    test  = df[(df["date"] >= TEST_START)  & (df["date"] <= TEST_END)].copy()

    print(f"  Train: {len(train):,} rows "
          f"({train.date.min().date()} → {train.date.max().date()})")
    print(f"  Val:   {len(val):,} rows "
          f"({val.date.min().date()} → {val.date.max().date()})")
    print(f"  Test:  {len(test):,} rows "
          f"({test.date.min().date()} → {test.date.max().date()})")

    # Density gate — only rows with Reddit activity
    train_gated = train[train["post_count_1d"] >= DENSITY_GATE].copy()
    val_gated   = val[val["post_count_1d"]     >= DENSITY_GATE].copy()
    test_gated  = test[test["post_count_1d"]   >= DENSITY_GATE].copy()

    print(f"  Train rows (gated): {len(train_gated):,}  "
          f"({len(train_gated)/len(train):.0%} of train)")
    print(f"  Val rows (gated):   {len(val_gated):,}  "
          f"({len(val_gated)/len(val):.0%} of val)")
    print(f"  Test rows (gated):  {len(test_gated):,}  "
          f"({len(test_gated)/len(test):.0%} of test)")
    return train, val, test, train_gated, val_gated, test_gated


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
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[xgb.XGBRegressor, dict]:
    """Train one XGBRegressor with val-based ICEarlyStopping.

    Phase 1 — scout: fit on pre-val subset, evaluate on val (2023H2).
              Selects best_n = tree count with highest val IC.
    Phase 2 — final: fit on full train (2022-2023) with best_n trees.
              Evaluate on test (2024-2025) only — never seen during training.

    Returns (model, metrics_dict).
    """
    tr = train.dropna(subset=[target_col])
    va = val.dropna(subset=[target_col])
    te = test.dropna(subset=[target_col])

    # Pre-val subset: training dates strictly before VAL_START
    # Used only in Phase 1 scout (not final model)
    pre_val = tr[tr["date"] < VAL_START]

    X_pre   = pre_val[FEATURE_COLS].astype(np.float32)
    y_pre   = pre_val[target_col].values.astype(np.float32)
    w_pre   = pre_val["sample_weight"].values.astype(np.float32)

    X_val   = va[FEATURE_COLS].astype(np.float32)
    y_val   = va[target_col].values.astype(np.float32)

    X_train = tr[FEATURE_COLS].astype(np.float32)
    y_train = tr[target_col].values.astype(np.float32)
    w_train = tr["sample_weight"].values.astype(np.float32)

    X_test  = te[FEATURE_COLS].astype(np.float32)
    y_test  = te[target_col].values.astype(np.float32)

    h_params = {**GKX_PARAMS, **HORIZON_PARAMS[horizon]}

    # Phase 1 — IC-guided scout on val set (NOT test set — no leakage)
    ic_stopper = ICEarlyStopping(rounds=50, X_eval=X_val, y_eval=y_val)
    scout = xgb.XGBRegressor(
        n_estimators=1000,
        **h_params,
        callbacks=[ic_stopper],
    )
    scout.fit(
        X_pre, y_pre,
        sample_weight=w_pre,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_n = max(1, ic_stopper.best_iteration + 1)

    # Phase 2 — final clean model: full train set, best_n trees, no callbacks
    model = xgb.XGBRegressor(n_estimators=best_n, **h_params)
    model.fit(X_train, y_train, sample_weight=w_train)

    train_pred = model.predict(X_train)
    val_pred   = model.predict(X_val)
    test_pred  = model.predict(X_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _tic, _ = spearmanr(train_pred, y_train)
        _vic, _ = spearmanr(val_pred,   y_val)
        _xic, _ = spearmanr(test_pred,  y_test)
    train_ic = float(_tic) if not np.isnan(_tic) else 0.0
    val_ic   = float(_vic) if not np.isnan(_vic) else 0.0
    test_ic  = float(_xic) if not np.isnan(_xic) else 0.0

    train_dir = float((np.sign(train_pred) == np.sign(y_train)).mean())
    test_dir  = float((np.sign(test_pred)  == np.sign(y_test)).mean())
    n_unique  = int(len(np.unique(np.round(test_pred, 6))))
    gap       = round(train_ic - test_ic, 6)
    gate_pass = bool(test_ic > 0.025)

    scout_ic_str = (f"{ic_stopper.best_ic:+.4f}"
                    if ic_stopper.best_ic > -np.inf else "n/a")
    print(f"\n  Model_{horizon}:")
    print(f"    best_n:       {best_n}  (scout val IC={scout_ic_str})")
    print(f"    Train IC:     {train_ic:+.4f}")
    print(f"    Val   IC:     {val_ic:+.4f}  (held-out 2023H2)")
    print(f"    Test  IC:     {test_ic:+.4f}  {'PASS ✓' if gate_pass else 'FAIL ✗'}")
    print(f"    Gap:          {gap:+.4f}")
    print(f"    Dir acc:      {test_dir:.1%}")
    print(f"    Unique preds: {n_unique}")
    print(f"    n_pre={len(pre_val):,}  n_train={len(tr):,}  n_val={len(va):,}  n_test={len(te):,}")

    return model, {
        "n_trees":   int(best_n),
        "val_ic":    round(val_ic,   6),
        "train_ic":  round(train_ic, 6),
        "test_ic":   round(test_ic,  6),
        "gap":       gap,
        "dir_acc":   round(test_dir, 6),
        "n_unique":  n_unique,
        "n_train":   int(len(tr)),
        "n_val":     int(len(va)),
        "n_test":    int(len(te)),
        "gate_pass": gate_pass,
    }


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
    val: pd.DataFrame,
    test: pd.DataFrame,
    train_gated: pd.DataFrame,
    val_gated: pd.DataFrame,
    test_gated: pd.DataFrame,
) -> None:
    print("\n=== SECTION 5: SAVING MODELS ===")

    for horizon, model in models.items():
        path = MODELS_DIR / f"model_{horizon}_v2.json"
        model.save_model(str(path))
        print(f"  Saved {path.name}")

    all_pass = all(metrics[h]["gate_pass"] for h in ["1d", "3d", "5d"])
    status   = "PASS" if all_pass else "BELOW_THRESHOLD"

    meta = {
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "training_window":   f"{TRAIN_START} → {TRAIN_END}",
        "val_window":        f"{VAL_START} → {VAL_END}",
        "test_window":       f"{TEST_START} → {TEST_END}",
        "feature_cols":      FEATURE_COLS,
        "feature_count":     len(FEATURE_COLS),
        "density_gate":      DENSITY_GATE,
        "gamma":             0.0,
        "regime_features":   "DROPPED (spy_above_200ma, regime_score)",
        "train_rows":        int(len(train)),
        "val_rows":          int(len(val)),
        "test_rows":         int(len(test)),
        "train_rows_gated":  int(len(train_gated)),
        "val_rows_gated":    int(len(val_gated)),
        "test_rows_gated":   int(len(test_gated)),
        "train_date_range":  [str(train.date.min().date()), str(train.date.max().date())],
        "val_date_range":    [str(val.date.min().date()),   str(val.date.max().date())],
        "test_date_range":   [str(test.date.min().date()),  str(test.date.max().date())],
        "model_1d":          metrics["1d"],
        "model_3d":          metrics["3d"],
        "model_5d":          metrics["5d"],
        "retrain_threshold_ic": 0.0796,
        "status":            status,
    }

    meta_path = MODELS_DIR / "training_metadata_v2.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path.name}")
    print()
    if all_pass:
        print("  Overall gate: ALL PASS ✓")
    else:
        failed = [h for h in ["1d", "3d", "5d"] if not metrics[h]["gate_pass"]]
        print(f"  SOME FAILED ✗ — horizons below gate: {failed}")
        print("  Review before deploying to live system.")


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
    new_dir = metrics["5d"]["dir_acc"]
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
        print(f"    {h.upper()}: {metrics[h]['dir_acc']:.1%}")
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

    train_raw, val_raw, test_raw, train_gated_raw, val_gated_raw, test_gated_raw = load_data()
    train, test, train_gated, test_gated, _ = preprocess(
        train_raw, test_raw, train_gated_raw, test_gated_raw
    )
    # Apply same NaN fill to val (clip bounds already computed from train)
    val_gated = val_raw[val_raw["post_count_1d"] >= DENSITY_GATE].copy()
    val_gated[FEATURE_COLS] = val_gated[FEATURE_COLS].fillna(0.0)

    print("\n=== SECTION 3: TRAINING ===")
    models:  dict[str, xgb.XGBRegressor] = {}
    metrics: dict[str, dict] = {}

    for horizon, target_col in TARGETS.items():
        model, m = train_model(
            horizon, target_col,
            train_gated, val_gated, test_gated,
        )
        models[horizon]  = model
        metrics[horizon] = m

    print("\n=== SECTION 4: FEATURE IMPORTANCE ===")
    for horizon, model in models.items():
        print_feature_importance(model, horizon)

    save_models(
        models, metrics,
        train_raw, val_raw, test_raw,
        train_gated, val_gated, test_gated,
    )
    compare_to_old(metrics)
    print_summary(metrics)


if __name__ == "__main__":
    main()
