#!/usr/bin/env python3
"""
Module: pipeline/05_validate_alpha.py
Purpose: Statistical validation of the Combined model's alpha.
         1. Permutation test (100 perms, p < 0.05 required)
         2. Bootstrap IC stability (50 boots, IC std < 0.03 required)

Phase: 1 — Research Pipeline
Input:  data/features/features.parquet
        models/registry/combined/
Output: reports/alpha_validation.json

Decision gate:
  - If permutation p-value >= 0.05: STOP — signal is spurious
  - If bootstrap IC std >= 0.03: flag instability warning

Usage:
    python pipeline/05_validate_alpha.py
    python pipeline/05_validate_alpha.py --debug
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
from config.thresholds import (
    N_PERMUTATIONS,
    N_BOOTSTRAP,
    PVALUE_THRESHOLD,
)
from pipeline.feature_schema import MARKET_FEATURES, REDDIT_FEATURES
from utils.logger import get_logger

log = get_logger(__name__)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    model,
    observed_ic: float,
    n_permutations: int = N_PERMUTATIONS,
    random_seed: int = 42,
) -> dict:
    """
    Permutation test: shuffle y, measure IC from re-predicted y_pred each time.

    Strategy: for each permutation, shuffle y (keeping X and model fixed),
    then compute the IC between the shuffled y and the model's predictions.
    This tests whether the model's IC could arise by chance.

    Note: We do NOT re-train for each permutation (too slow, and the goal
    is to test whether the label ordering matters, not the model itself).
    We compare observed IC against the null distribution of IC(model_preds, shuffled_y).

    Args:
        X: Feature matrix (test set)
        y: True labels (test set)
        model: Trained model with .predict()
        observed_ic: IC from actual (un-shuffled) predictions
        n_permutations: Number of shuffles
        random_seed: NumPy RNG seed

    Returns:
        dict with p_value, null distribution stats, verdict
    """
    rng = np.random.default_rng(random_seed)
    preds = model.predict(X)

    null_ics = []
    for _ in range(n_permutations):
        shuffled_y = rng.permutation(y)
        ic = compute_ic(shuffled_y, preds)
        null_ics.append(ic)

    null_ics = np.array(null_ics)
    # One-sided: what fraction of null ICs >= observed IC?
    p_value = float(np.mean(null_ics >= observed_ic))

    result = {
        "observed_ic": round(observed_ic, 6),
        "n_permutations": n_permutations,
        "null_ic_mean": round(float(np.mean(null_ics)), 6),
        "null_ic_std": round(float(np.std(null_ics)), 6),
        "null_ic_p95": round(float(np.percentile(null_ics, 95)), 6),
        "p_value": round(p_value, 4),
        "p_value_threshold": PVALUE_THRESHOLD,
        "verdict": "SIGNIFICANT" if p_value < PVALUE_THRESHOLD else "SPURIOUS",
    }

    log.info(
        "permutation_test_complete",
        p_value=round(p_value, 4),
        observed_ic=round(observed_ic, 4),
        null_p95=round(float(np.percentile(null_ics, 95)), 4),
        verdict=result["verdict"],
    )
    return result


# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

def run_bootstrap_stability(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_bootstrap: int = N_BOOTSTRAP,
    random_seed: int = 42,
) -> dict:
    """
    Bootstrap IC stability: resample test set with replacement, compute IC each time.
    If IC std >= 0.03 → signal is unstable.

    Args:
        X: Feature matrix
        y: True labels
        model: Trained model
        n_bootstrap: Number of bootstrap samples
        random_seed: RNG seed

    Returns:
        dict with mean IC, std IC, CI, verdict
    """
    rng = np.random.default_rng(random_seed)
    preds = model.predict(X)
    n = len(y)

    boot_ics = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ic = compute_ic(y[idx], preds[idx])
        boot_ics.append(ic)

    boot_ics = np.array(boot_ics)

    ic_mean = float(np.mean(boot_ics))
    ic_std = float(np.std(boot_ics))
    ci_low = float(np.percentile(boot_ics, 2.5))
    ci_high = float(np.percentile(boot_ics, 97.5))

    STABILITY_THRESHOLD = 0.03
    result = {
        "n_bootstrap": n_bootstrap,
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "ic_ci_low": round(ci_low, 6),
        "ic_ci_high": round(ci_high, 6),
        "stability_threshold": STABILITY_THRESHOLD,
        "verdict": "STABLE" if ic_std < STABILITY_THRESHOLD else "UNSTABLE",
        "boot_ic_values": [round(v, 4) for v in boot_ics.tolist()],
    }

    log.info(
        "bootstrap_complete",
        ic_mean=round(ic_mean, 4),
        ic_std=round(ic_std, 4),
        ci=f"[{ci_low:.4f}, {ci_high:.4f}]",
        verdict=result["verdict"],
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — Statistical Alpha Validation")
    parser.add_argument("--debug", action="store_true",
                        help="Use 5 tickers, 10 permutations, 5 bootstraps")
    parser.add_argument("--model", default="combined",
                        help="Model to validate (combined/market/reddit)")
    args = parser.parse_args()

    # Load model
    model_dir = MODELS_DIR / args.model
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
    test_df = df[df["split"] == "test"].dropna(subset=["target_return_5d"]).copy()

    if test_df.empty:
        print("ERROR: No test data.")
        sys.exit(1)

    if args.debug:
        tickers = test_df["ticker"].unique()[:5]
        test_df = test_df[test_df["ticker"].isin(tickers)]
        n_perm = 10
        n_boot = 5
        log.info("debug_mode", tickers=list(tickers), n_perm=n_perm, n_boot=n_boot)
    else:
        n_perm = N_PERMUTATIONS
        n_boot = N_BOOTSTRAP

    log.info("validation_start", model=args.model, n_test=len(test_df))

    # Build feature matrix
    missing = [c for c in feat_cols if c not in test_df.columns]
    if missing:
        log.warning("missing_features", cols=missing)
        feat_cols = [c for c in feat_cols if c in test_df.columns]

    X_test = test_df[feat_cols].copy()
    if args.model == "reddit":
        X_test = X_test.fillna(0)
    else:
        X_test = X_test.fillna(X_test.median())

    X_np = X_test.values
    y_np = test_df["target_return_5d"].values

    # Observed IC
    preds = model.predict(X_np)
    observed_ic = compute_ic(y_np, preds)
    log.info("observed_ic", ic=round(observed_ic, 4))

    # Run permutation test
    perm_result = run_permutation_test(
        X_np, y_np, model, observed_ic, n_permutations=n_perm
    )

    # Run bootstrap stability
    boot_result = run_bootstrap_stability(
        X_np, y_np, model, n_bootstrap=n_boot
    )

    # Compose report
    report = {
        "run_date": pd.Timestamp.now().isoformat(),
        "model": args.model,
        "n_test_rows": len(test_df),
        "observed_ic": round(observed_ic, 6),
        "permutation_test": {k: v for k, v in perm_result.items()},
        "bootstrap_stability": {k: v for k, v in boot_result.items()
                                if k != "boot_ic_values"},
        "overall_verdict": (
            "PROCEED"
            if perm_result["verdict"] == "SIGNIFICANT"
            and boot_result["verdict"] == "STABLE"
            else "FLAG"
        ),
        "flags": [],
    }

    if perm_result["verdict"] == "SPURIOUS":
        report["flags"].append(
            f"Permutation p-value={perm_result['p_value']:.4f} >= {PVALUE_THRESHOLD} — signal may be spurious"
        )
    if boot_result["verdict"] == "UNSTABLE":
        report["flags"].append(
            f"Bootstrap IC std={boot_result['ic_std']:.4f} >= 0.03 — signal is unstable"
        )

    report_path = REPORTS_DIR / "alpha_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("alpha_validation_saved", path=str(report_path))

    # Console
    print("\n=== ALPHA VALIDATION ===")
    print(f"  Model:            {args.model}")
    print(f"  Observed IC:      {observed_ic:.4f}")
    print()
    print("  Permutation Test:")
    print(f"    p-value:        {perm_result['p_value']:.4f}  (threshold: {PVALUE_THRESHOLD})")
    print(f"    Null IC p95:    {perm_result['null_ic_p95']:.4f}")
    print(f"    Verdict:        {perm_result['verdict']}")
    print()
    print("  Bootstrap Stability:")
    print(f"    IC mean:        {boot_result['ic_mean']:.4f}")
    print(f"    IC std:         {boot_result['ic_std']:.4f}  (threshold: 0.03)")
    print(f"    IC 95% CI:      [{boot_result['ic_ci_low']:.4f}, {boot_result['ic_ci_high']:.4f}]")
    print(f"    Verdict:        {boot_result['verdict']}")
    print()
    print(f"  OVERALL: {report['overall_verdict']}")
    for flag in report["flags"]:
        print(f"  ⚠️  {flag}")

    if perm_result["verdict"] == "SPURIOUS":
        print("\n  🚨 STOP: Permutation p-value >= 0.05. Signal may be spurious.")
        print("          Per spec: Report findings and do not proceed to live trading.")

    print(f"\n  Report saved to {report_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
