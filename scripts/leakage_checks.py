#!/usr/bin/env python3
"""
leakage_checks.py — 7 data-leakage and integrity checks before every retrain.

Usage:
    python scripts/leakage_checks.py
    python scripts/leakage_checks.py --path data/features/features_v2.parquet

Exit 0: all checks PASS (WARNs do not count as failures)
Exit 1: at least one check FAIL
"""

import argparse
import json
import random
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "post_count_1d",       "abnormal_attention_1d",
    "total_comments_1d",   "vader_sentiment_1d",
    "sentiment_extremity", "sentiment_accel",
    "volume",              "relative_volume",
    "returns_1d",          "returns_20d",
    "rsi_14",              "news_sentiment_1d",
    "vix_percentile",      "vix_x_volume",
    "spy_above_200ma",     "regime_score",
    "dist_from_20ma_pct",  "pead_proxy",
]

TRAIN_END_DATE   = "2023-12-31"
TEST_START_DATE  = "2024-01-01"
FUTURE_HORIZON   = 5          # trading days
CALENDAR_BUFFER  = 7          # ~5 trading days in calendar days
LEAKAGE_CORR_HI  = 0.85
LEAKAGE_CORR_MED = 0.50


# ── CHECK 1 ───────────────────────────────────────────────────────────────────
def check_1_no_future_timestamps(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    No row should have target_return_5d filled when its date is within
    CALENDAR_BUFFER days of today (t+5 prices can't exist yet).
    """
    today   = pd.Timestamp.today().normalize()
    cutoff  = today - pd.Timedelta(days=CALENDAR_BUFFER)

    # rows with recent date AND a non-null 5-day target
    bad = df[
        (df["date"] >= cutoff) &
        df["target_return_5d"].notna() &
        (df["target_return_5d"] != 0.0)
    ]

    if bad.empty:
        return "PASS", []

    msgs = []
    for _, row in bad.iterrows():
        msgs.append(
            f"  {row['ticker']} @ {row['date']}  "
            f"target_5d={row['target_return_5d']:+.4f}"
        )
    return "FAIL", msgs


# ── CHECK 2 ───────────────────────────────────────────────────────────────────
def check_2_feature_stability(df: pd.DataFrame, n_samples: int = 5) -> tuple[str, list[str]]:
    """
    Verify returns_1d and returns_20d are internally consistent with the
    stored close prices. Checks that features were computed from history only.

    For each (ticker, row_idx) sample:
      - stored returns_1d  must match  (close[idx] - close[idx-1]) / close[idx-1]
      - stored returns_20d must match  (close[idx] - close[idx-20]) / close[idx-20]
    Tolerance: 0.5% (0.005) to allow minor rounding in the feature builder.
    """
    if "close" not in df.columns:
        return "WARN", ["  close column not in parquet — skipping stability check"]

    tickers = [
        t for t in df["ticker"].unique()
        if len(df[df["ticker"] == t]) > 25
    ]
    if not tickers:
        return "WARN", ["  No ticker has >25 rows — skipping stability check"]

    sample_pool = tickers[:50]
    chosen = random.sample(sample_pool, min(n_samples, len(sample_pool)))

    failures: list[str] = []
    for ticker in chosen:
        t_df = (
            df[df["ticker"] == ticker]
            .sort_values("date")
            .reset_index(drop=True)
        )

        # pick a mid-range index
        idx = random.randint(20, len(t_df) - 1)
        row = t_df.iloc[idx]

        close_now = float(row["close"])

        # --- returns_1d ---
        if "returns_1d" in df.columns:
            close_prev1 = float(t_df.iloc[idx - 1]["close"])
            if close_prev1 != 0:
                computed_r1 = (close_now - close_prev1) / close_prev1
                stored_r1   = float(row["returns_1d"])
                if abs(stored_r1 - computed_r1) > 0.005:
                    failures.append(
                        f"  {ticker} @ {row['date']}  returns_1d: "
                        f"stored={stored_r1:.5f}  computed={computed_r1:.5f}"
                    )

        # --- returns_20d ---
        if "returns_20d" in df.columns:
            close_prev20 = float(t_df.iloc[idx - 20]["close"])
            if close_prev20 != 0:
                computed_r20 = (close_now - close_prev20) / close_prev20
                stored_r20   = float(row["returns_20d"])
                if abs(stored_r20 - computed_r20) > 0.005:
                    failures.append(
                        f"  {ticker} @ {row['date']}  returns_20d: "
                        f"stored={stored_r20:.5f}  computed={computed_r20:.5f}"
                    )

    if failures:
        return "FAIL", failures
    return "PASS", [f"  Spot-checked {len(chosen)} tickers — all within tolerance"]


# ── CHECK 3 ───────────────────────────────────────────────────────────────────
def check_3_purge_embargo_audit(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Check that each fold in experiments/walk_forward/results.json has a clean
    gap of at least PURGE_DAYS + EMBARGO_DAYS trading days between train_end
    and test_start.  Approximated as 14+ calendar days (10 trading days × 1.4).
    """
    results_path = BASE / "experiments" / "walk_forward" / "results.json"
    if not results_path.exists():
        return "WARN", [
            "  experiments/walk_forward/results.json not found — "
            "run walk_forward_validation.py first"
        ]

    with open(results_path) as f:
        wf = json.load(f)

    purge   = wf.get("purge_days",   5)
    embargo = wf.get("embargo_days", 5)
    min_gap_td = int((purge + embargo) * 1.4)  # calendar days

    violations: list[str] = []
    for fold in wf.get("folds", []):
        train_end  = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        gap_days   = (test_start - train_end).days

        if gap_days < min_gap_td:
            violations.append(
                f"  Fold {fold['fold']}: gap={gap_days}d "
                f"< required {min_gap_td}d  "
                f"(train_end={fold['train_end']}  "
                f"test_start={fold['test_start']})"
            )

    if violations:
        return "FAIL", violations

    n_folds = len(wf.get("folds", []))
    return "PASS", [f"  {n_folds} folds checked — all gaps ≥ {min_gap_td} calendar days"]


# ── CHECK 4 ───────────────────────────────────────────────────────────────────
def check_4_target_leakage_score(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Spearman correlation of each feature vs target_return_5d.
    HIGH RISK  if |corr| > 0.85 → FAIL
    MODERATE   if |corr| > 0.50 → WARN
    """
    if "target_return_5d" not in df.columns:
        return "WARN", ["  target_return_5d column missing — skipping leakage score"]

    target = df["target_return_5d"].dropna()
    if len(target) < 50:
        return "WARN", ["  Fewer than 50 non-null target rows — skipping leakage score"]

    high_risk: list[str] = []
    moderate:  list[str] = []

    for feat in FEATURE_COLS:
        if feat not in df.columns:
            continue
        aligned = df.loc[target.index, feat]
        valid   = aligned.notna() & target.notna()
        if valid.sum() < 30:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw, _ = spearmanr(aligned[valid], target[valid])
        corr = float(raw) if not np.isnan(float(raw)) else 0.0

        if abs(corr) > LEAKAGE_CORR_HI:
            high_risk.append(
                f"  HIGH RISK   {feat:<28} ρ={corr:+.4f}"
            )
        elif abs(corr) > LEAKAGE_CORR_MED:
            moderate.append(
                f"  MODERATE    {feat:<28} ρ={corr:+.4f}"
            )

    msgs = high_risk + moderate
    if not msgs:
        msgs = [f"  All {len(FEATURE_COLS)} features within safe correlation range"]

    status = "FAIL" if high_risk else ("WARN" if moderate else "PASS")
    return status, msgs


# ── CHECK 5 ───────────────────────────────────────────────────────────────────
def check_5_normalization_check(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    Heuristic check for suspicious min-max or z-score normalization applied
    to raw continuous features before they were stored.
    WARN only — does not fail the suite.
    """
    suspects = [
        "returns_1d", "returns_20d", "rsi_14",
        "relative_volume", "volume",
    ]
    warnings_out: list[str] = []

    for feat in suspects:
        if feat not in df.columns:
            continue
        vals = df[feat].dropna()
        if len(vals) < 100:
            continue

        mn   = float(vals.min())
        mx   = float(vals.max())
        mean = float(vals.mean())
        std  = float(vals.std())

        # Looks min-max scaled: range in [0,1], mean away from extremes
        if mn >= -0.01 and mx <= 1.01 and 0.1 < mean < 0.9:
            warnings_out.append(
                f"  {feat}: range [{mn:.3f},{mx:.3f}] looks min-max scaled "
                f"(mean={mean:.3f})"
            )
        # Looks z-scored: mean≈0, std≈1, tight range
        elif abs(mean) < 0.05 and 0.9 < std < 1.1 and abs(mn) < 5 and abs(mx) < 5:
            warnings_out.append(
                f"  {feat}: mean={mean:.3f} std={std:.3f} looks z-scored"
            )

    if not warnings_out:
        return "PASS", [f"  {len(suspects)} features checked — no normalization artifacts"]
    return "WARN", warnings_out


# ── CHECK 6 ───────────────────────────────────────────────────────────────────
def check_6_duplicate_rows(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    1. No (ticker, date) duplicate rows.
    2. No (ticker, date) key appears in both train split and test split.
    """
    msgs: list[str] = []

    # --- Exact duplicates ---
    dup_mask  = df.duplicated(subset=["ticker", "date"], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        examples = (
            df[dup_mask][["ticker", "date"]]
            .drop_duplicates()
            .head(5)
        )
        msgs.append(f"  {dup_count} duplicate (ticker,date) rows found")
        for _, row in examples.iterrows():
            msgs.append(f"    {row['ticker']} @ {row['date']}")

    # --- Train / test overlap ---
    cutoff    = pd.Timestamp(TRAIN_END_DATE)
    t_start   = pd.Timestamp(TEST_START_DATE)
    train_set = set(
        zip(df[df["date"] <= cutoff]["ticker"],
            df[df["date"] <= cutoff]["date"].astype(str))
    )
    test_set  = set(
        zip(df[df["date"] >= t_start]["ticker"],
            df[df["date"] >= t_start]["date"].astype(str))
    )
    overlap   = train_set & test_set
    if overlap:
        msgs.append(f"  {len(overlap)} (ticker,date) keys in BOTH train and test splits")
        for item in list(overlap)[:5]:
            msgs.append(f"    {item[0]} @ {item[1]}")

    if msgs:
        return "FAIL", msgs

    n_train = int((df["date"] <= cutoff).sum())
    n_test  = int((df["date"] >= t_start).sum())
    return "PASS", [
        f"  No duplicates  |  train={n_train:,} rows  test={n_test:,} rows  "
        f"no overlap"
    ]


# ── CHECK 7 ───────────────────────────────────────────────────────────────────
def check_7_sentiment_timestamp_audit(df: pd.DataFrame) -> tuple[str, list[str]]:
    """
    1. No live post row (post_count_1d > 0) within CALENDAR_BUFFER days of
       today should have a filled target_return_5d (future price unavailable).
    2. sentiment_accel should be ~0 for the first 3 dates per ticker
       (no prior day to compute acceleration from).
    """
    msgs:  list[str] = []
    today  = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=CALENDAR_BUFFER)

    # --- 7a: future prices for live posts ---
    bad_live = df[
        (df["date"] >= cutoff) &
        (df.get("post_count_1d", pd.Series(0, index=df.index)) > 0) &
        df["target_return_5d"].notna() &
        (df["target_return_5d"] != 0.0)
    ]
    if not bad_live.empty:
        msgs.append(
            f"  7a: {len(bad_live)} recent live-post rows with target_return_5d filled"
        )
        for _, row in bad_live.head(5).iterrows():
            msgs.append(
                f"    {row['ticker']} @ {row['date']}  "
                f"post_count={row.get('post_count_1d', '?')}  "
                f"target_5d={row['target_return_5d']:+.4f}"
            )

    # --- 7b: sentiment_accel at each mid-dataset ticker's 3 earliest dates ---
    # Established tickers (present from dataset start) have pre-history that
    # produces non-zero sentiment_accel at 2019-01-03 — that is correct.
    # Only flag tickers that newly appear >= 60 days after the dataset start,
    # where no prior sentiment data should exist.
    if "sentiment_accel" in df.columns:
        dataset_start    = df["date"].min()
        new_ticker_cutoff = dataset_start + pd.Timedelta(days=60)

        accel_violations: list[str] = []
        for ticker, group in df.groupby("ticker"):
            first_date = group["date"].min()
            if first_date < new_ticker_cutoff:
                continue   # established ticker — pre-history expected

            early = group.sort_values("date").head(3)
            bad   = early[early["sentiment_accel"].abs() > 0.001]
            if not bad.empty:
                for _, row in bad.iterrows():
                    accel_violations.append(
                        f"    {ticker} @ {row['date']}  "
                        f"sentiment_accel={row['sentiment_accel']:.5f}  "
                        f"(ticker first appeared {first_date.date()})"
                    )
        if accel_violations:
            msgs.append(
                f"  7b WARN: sentiment_accel non-zero at new-ticker first-3-date rows "
                f"({len(accel_violations)} cases — may be pre-IPO Reddit data, investigate)"
            )
            msgs.extend(accel_violations[:10])

    # 7a violations are FAIL; 7b are WARN (pre-IPO social data is expected)
    has_7a = bad_live is not None and not bad_live.empty
    if has_7a:
        return "FAIL", msgs
    if msgs:
        return "WARN", msgs

    return "PASS", [
        "  No future-price leakage in live rows  |  "
        "sentiment_accel clean at each new ticker's start"
    ]


# ── Runner ────────────────────────────────────────────────────────────────────
def run_all(path: Path) -> bool:
    """
    Run all 7 checks.  Returns True if suite PASSES (no FAIL status).
    """
    print(f"\nLoading  {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  {len(df):,} rows  |  {df['ticker'].nunique()} tickers  "
          f"|  {df['date'].min().date()} → {df['date'].max().date()}\n")

    checks = [
        ("CHECK 1  No future timestamps",         check_1_no_future_timestamps),
        ("CHECK 2  Feature stability",             check_2_feature_stability),
        ("CHECK 3  Purge/embargo audit",           check_3_purge_embargo_audit),
        ("CHECK 4  Target leakage score",          check_4_target_leakage_score),
        ("CHECK 5  Normalization check",           check_5_normalization_check),
        ("CHECK 6  Duplicate rows",                check_6_duplicate_rows),
        ("CHECK 7  Sentiment timestamp audit",     check_7_sentiment_timestamp_audit),
    ]

    results: list[tuple[str, str]] = []
    any_fail = False

    for label, fn in checks:
        try:
            status, msgs = fn(df)
        except Exception as exc:
            status = "ERROR"
            msgs   = [f"  Exception: {exc}"]

        if status == "FAIL" or status == "ERROR":
            any_fail = True

        marker = {
            "PASS":  "✓",
            "WARN":  "⚠",
            "FAIL":  "✗",
            "ERROR": "!",
        }.get(status, "?")

        print(f"[{status:<4}] {marker} {label}")
        for msg in msgs:
            print(msg)
        print()

        results.append((label, status))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for _, s in results if s == "PASS")
    n_warn = sum(1 for _, s in results if s == "WARN")
    n_fail = sum(1 for _, s in results if s in ("FAIL", "ERROR"))

    print("─" * 56)
    print(f"OVERALL: {'PASS ✓' if not any_fail else 'FAIL ✗'}  "
          f"(pass={n_pass}  warn={n_warn}  fail={n_fail})")
    print("─" * 56)

    return not any_fail


def main() -> None:
    parser = argparse.ArgumentParser(description="RSSS leakage checks")
    parser.add_argument(
        "--path",
        default=str(BASE / "data" / "features" / "features_v2.parquet"),
        help="Path to feature parquet (default: features_v2.parquet)",
    )
    args = parser.parse_args()

    passed = run_all(Path(args.path))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
