#!/usr/bin/env python3
"""
Module: pipeline/06_feature_importance.py
Purpose: SHAP analysis for Combined model + ablation table (6 feature subsets).
         Use TreeExplainer (NOT model.feature_importances_).

Phase: 1 — Research Pipeline
Input:  data/features/features.parquet
        models/registry/combined/
Output: reports/feature_importance.json
        reports/feature_importance.html
        reports/shap_summary.png  (if matplotlib available)

Usage:
    python pipeline/06_feature_importance.py
    python pipeline/06_feature_importance.py --debug
Last modified: 2026-06-11
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.shared.metrics import compute_ic

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import FEATURES_PARQUET, MODELS_DIR, REPORTS_DIR
from pipeline.feature_schema import (
    MARKET_FEATURES,
    REDDIT_FEATURES,
    TARGET_COL,
    ABLATION_SUBSETS,
)
from utils.logger import get_logger

log = get_logger(__name__)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Run: pip install shap")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed.")
    sys.exit(1)

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Train a fast ablation model
# ---------------------------------------------------------------------------

def train_ablation_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBRegressor:
    """Train a lightweight XGBoost for ablation (fewer trees, fast)."""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, verbose=False)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — Feature Importance & SHAP")
    parser.add_argument("--debug", action="store_true",
                        help="Use 2 tickers, small SHAP sample")
    args = parser.parse_args()

    # Load model
    model_dir = MODELS_DIR / "combined"
    pkl_path = model_dir / "model.pkl"
    meta_path = model_dir / "metadata.json"

    if not pkl_path.exists():
        print(f"ERROR: Model not found at {pkl_path}. Run 03_train_models.py first.")
        sys.exit(1)

    model = joblib.load(pkl_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]

    # Load features
    if not FEATURES_PARQUET.exists():
        print(f"ERROR: {FEATURES_PARQUET} not found.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PARQUET)
    train_df = df[df["split"] == "train"].dropna(subset=[TARGET_COL]).copy()
    test_df = df[df["split"] == "test"].dropna(subset=[TARGET_COL]).copy()

    if args.debug:
        tickers = test_df["ticker"].unique()[:2]
        train_df = train_df[train_df["ticker"].isin(tickers)]
        test_df = test_df[test_df["ticker"].isin(tickers)]
        log.info("debug_mode", tickers=list(tickers))

    log.info("fi_start", train_rows=len(train_df), test_rows=len(test_df))

    # Prepare feature matrices
    def prep_features(d: pd.DataFrame, cols: list[str], fill_reddit: bool = False) -> pd.DataFrame:
        available = [c for c in cols if c in d.columns]
        X = d[available].copy()
        if fill_reddit:
            X = X.fillna(0)
        else:
            X = X.fillna(X.median())
        return X

    # -----------------------------------------------------------------------
    # SHAP analysis on Combined model (use test set, max 2000 rows for speed)
    # -----------------------------------------------------------------------
    available_feat_cols = [c for c in feat_cols if c in test_df.columns]
    X_shap = prep_features(test_df, available_feat_cols)
    shap_sample = X_shap.sample(min(2000, len(X_shap)), random_state=42)

    log.info("running_shap", n_samples=len(shap_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)

    # Mean absolute SHAP per feature
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=available_feat_cols,
    ).sort_values(ascending=False)

    shap_ranking = {
        feat: round(float(val), 6)
        for feat, val in mean_abs_shap.items()
    }

    log.info("shap_complete", top_3=mean_abs_shap.head(3).to_dict())

    # Save SHAP plot if matplotlib available
    shap_plot_saved = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(20, len(mean_abs_shap))
        top_feats = mean_abs_shap.head(top_n)
        ax.barh(top_feats.index[::-1], top_feats.values[::-1])
        ax.set_xlabel("Mean |SHAP| value")
        ax.set_title("Feature Importance (SHAP) — Combined Model")
        plt.tight_layout()
        plot_path = REPORTS_DIR / "shap_summary.png"
        plt.savefig(str(plot_path), dpi=120, bbox_inches="tight")
        plt.close()
        shap_plot_saved = True
        log.info("shap_plot_saved", path=str(plot_path))
    except Exception as e:
        log.warning("shap_plot_failed", error=str(e))

    # -----------------------------------------------------------------------
    # Ablation table — 6 subsets, each trained on train_df, IC on test_df
    # -----------------------------------------------------------------------
    ablation_results = {}

    for subset_name, feature_list in ABLATION_SUBSETS.items():
        available = [c for c in feature_list if c in train_df.columns]
        if not available:
            log.warning("ablation_subset_empty", subset=subset_name)
            ablation_results[subset_name] = {"ic": None, "n_features": 0, "features": []}
            continue

        fill_reddit = subset_name.startswith("reddit")
        X_tr = prep_features(train_df, available, fill_reddit=fill_reddit)
        X_te = prep_features(test_df, available, fill_reddit=fill_reddit)
        y_tr = train_df[TARGET_COL]
        y_te = test_df[TARGET_COL]

        ablation_model = train_ablation_model(X_tr, y_tr)
        preds = ablation_model.predict(X_te)
        ic = compute_ic(y_te.values, preds)

        ablation_results[subset_name] = {
            "ic": round(ic, 6),
            "n_features": len(available),
            "features": available,
        }
        log.info("ablation", subset=subset_name, ic=round(ic, 4), n_features=len(available))

    # -----------------------------------------------------------------------
    # Compose report
    # -----------------------------------------------------------------------
    # Determine which feature groups are most impactful
    reddit_feats_in_model = [f for f in available_feat_cols if f in REDDIT_FEATURES]
    market_feats_in_model = [f for f in available_feat_cols if f in MARKET_FEATURES]

    top_reddit = {
        f: shap_ranking[f]
        for f in reddit_feats_in_model
        if f in shap_ranking
    }
    top_market = {
        f: shap_ranking[f]
        for f in market_feats_in_model
        if f in shap_ranking
    }

    # Sort by SHAP value
    top_reddit = dict(sorted(top_reddit.items(), key=lambda x: -x[1]))
    top_market = dict(sorted(top_market.items(), key=lambda x: -x[1]))

    report = {
        "run_date": pd.Timestamp.now().isoformat(),
        "model": "combined",
        "n_features": len(available_feat_cols),
        "shap_ranking": shap_ranking,
        "top_5_reddit_features": dict(list(top_reddit.items())[:5]),
        "top_5_market_features": dict(list(top_market.items())[:5]),
        "ablation_table": ablation_results,
        "shap_plot_saved": shap_plot_saved,
    }

    report_path = REPORTS_DIR / "feature_importance.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("feature_importance_saved", path=str(report_path))

    # HTML report
    ablation_rows = ""
    for name, r in ablation_results.items():
        ic_str = f"{r['ic']:.4f}" if r["ic"] is not None else "N/A"
        ablation_rows += f"<tr><td>{name}</td><td>{r['n_features']}</td><td>{ic_str}</td></tr>"

    shap_rows = ""
    for feat, val in list(shap_ranking.items())[:20]:
        group = "reddit" if feat in REDDIT_FEATURES else "market"
        shap_rows += f"<tr><td>{feat}</td><td>{group}</td><td>{val:.4f}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><title>RSSS Feature Importance</title>
<style>
  body {{ font-family: monospace; padding: 20px; }}
  table {{ border-collapse: collapse; margin-bottom: 30px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
  th {{ background: #f0f0f0; }}
</style></head><body>
<h2>RSSS Phase 1 — Feature Importance & Ablation</h2>
<h3>Top 20 Features by Mean |SHAP|</h3>
<table>
  <tr><th>Feature</th><th>Group</th><th>Mean |SHAP|</th></tr>
  {shap_rows}
</table>
<h3>Ablation Table (6 Subsets)</h3>
<table>
  <tr><th>Subset</th><th>N Features</th><th>Test IC</th></tr>
  {ablation_rows}
</table>
{"<img src='shap_summary.png' width='700'/>" if shap_plot_saved else ""}
<p><small>Generated by pipeline/06_feature_importance.py</small></p>
</body></html>"""

    html_path = REPORTS_DIR / "feature_importance.html"
    html_path.write_text(html)
    log.info("feature_importance_html_saved", path=str(html_path))

    # Console summary
    print("\n=== FEATURE IMPORTANCE (SHAP) ===")
    print("  Top 10 features by mean |SHAP|:")
    for feat, val in list(shap_ranking.items())[:10]:
        group = "(reddit)" if feat in REDDIT_FEATURES else "(market)"
        print(f"    {feat:35s} {val:.4f}  {group}")
    print()
    print("  Ablation IC summary:")
    for name, r in ablation_results.items():
        ic_str = f"{r['ic']:.4f}" if r["ic"] is not None else "N/A"
        print(f"    {name:25s} IC = {ic_str}  ({r['n_features']} features)")
    print()
    print(f"  Reports saved to {REPORTS_DIR}/")
    print("=" * 40)


if __name__ == "__main__":
    main()
