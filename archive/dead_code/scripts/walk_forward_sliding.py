#!/usr/bin/env python3
"""
walk_forward_sliding.py — RSSS v2 SLIDING-window walk-forward.

Reuses all training, simulation, and reporting logic from
walk_forward_validation.py.  Only fold generation changes:
train window has a FIXED length (4 years) that slides forward
instead of expanding from the origin.

Usage:
    python scripts/walk_forward_sliding.py
    python scripts/walk_forward_sliding.py --folds 3

Results: experiments/walk_forward_sliding/results.json
Models:  experiments/walk_forward_sliding/fold_models/fold_N_model_{hz}.json
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=UserWarning)

# ── Import shared logic from expanding-window WFV ─────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent))

import scripts.walk_forward_validation as _wfv

# Re-export constants so callers see them here
FEATURE_COLS   = _wfv.FEATURE_COLS
TARGETS        = _wfv.TARGETS
GKX_PARAMS     = _wfv.GKX_PARAMS
HORIZON_PARAMS = _wfv.HORIZON_PARAMS
DENSITY_GATE   = _wfv.DENSITY_GATE
WF_START       = _wfv.WF_START
WF_END         = _wfv.WF_END

ICEarlyStopping          = _wfv.ICEarlyStopping
train_fold_models        = _wfv.train_fold_models
simulate                 = _wfv.simulate
aggregate_results        = _wfv.aggregate_results
deflated_sharpe_confidence = _wfv.deflated_sharpe_confidence
print_report             = _wfv.print_report
get_regime_label         = _wfv.get_regime_label
IS_LOOKBACK              = _wfv.IS_LOOKBACK

# ── Sliding-window constants ──────────────────────────────────────────────────
SLIDING_TRAIN_YEARS  = 4
SLIDING_VAL_MONTHS   = 6
SLIDING_TEST_MONTHS  = 6
SLIDING_STEP_MONTHS  = 6
PURGE_DAYS           = 5
EMBARGO_DAYS         = 21   # longer for sliding (1 month buffer)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).resolve().parent.parent
FEAT_PATH     = BASE / "data" / "features" / "features_v2.parquet"
SL_DIR        = BASE / "experiments" / "walk_forward_sliding"
SL_FOLD_MODELS = SL_DIR / "fold_models"
SL_RESULTS    = SL_DIR / "results.json"
EXP_RESULTS   = BASE / "experiments" / "walk_forward" / "results.json"


# ── Sliding fold generation ───────────────────────────────────────────────────
def generate_sliding_folds(all_dates: list[str]) -> list[dict]:
    """
    Fixed-length (4-year) training windows that slide forward by 6 months.

    Timeline per fold:
      [train_start ─── 4yr ─── train_end] [─ 6mo val ─] [21d gap] [─ 6mo test ─]

    The val window creates a clean buffer before the embargo gap,
    ensuring no label overlap at the train/test boundary.
    """
    from dateutil.relativedelta import relativedelta

    end_ts = pd.Timestamp(all_dates[-1])
    anchor = pd.Timestamp(all_dates[0])
    folds  = []
    fold_n = 1

    def nearest_on_or_after(ts: pd.Timestamp) -> str:
        s = ts.strftime("%Y-%m-%d")
        for d in all_dates:
            if d >= s:
                return d
        return all_dates[-1]

    while True:
        train_start = anchor
        train_end   = anchor + relativedelta(years=SLIDING_TRAIN_YEARS)
        val_end     = train_end + relativedelta(months=SLIDING_VAL_MONTHS)
        test_start  = val_end   + pd.Timedelta(days=EMBARGO_DAYS)
        test_end    = test_start + relativedelta(months=SLIDING_TEST_MONTHS)

        if test_end > end_ts:
            break

        ts_str  = nearest_on_or_after(train_start)
        te_str  = nearest_on_or_after(train_end)
        ve_str  = nearest_on_or_after(val_end)
        tst_str = nearest_on_or_after(test_start)
        ted_str = nearest_on_or_after(test_end)

        folds.append({
            "fold":        fold_n,
            "train_start": ts_str,
            "train_end":   te_str,
            "val_end":     ve_str,
            "test_start":  tst_str,
            "test_end":    ted_str,
            "regime":      get_regime_label(tst_str),
        })

        anchor += relativedelta(months=SLIDING_STEP_MONTHS)
        fold_n  += 1

    return folds


# ── Comparison helper ─────────────────────────────────────────────────────────
def _print_comparison(exp_agg: dict, sl_agg: dict) -> None:
    def _pick(a: float, b: float, higher_better: bool = True) -> str:
        if higher_better:
            return "Expanding" if a > b else "Sliding"
        return "Expanding" if a < b else "Sliding"

    W = 56
    print()
    print("═" * W)
    print("EXPANDING vs SLIDING WINDOW COMPARISON")
    print("═" * W)
    hdr = f"  {'Metric':<26}  {'Expanding':>9}  {'Sliding':>9}  {'Better':>10}"
    print(hdr)
    print("  " + "─" * 52)

    rows = [
        ("Pooled Sharpe",         exp_agg["pooled_sharpe"],       sl_agg["pooled_sharpe"],       True),
        ("% Profitable folds",    exp_agg["pct_positive_folds"],  sl_agg["pct_positive_folds"],  True),
        ("WFE ratio",             exp_agg["wfe"],                  sl_agg["wfe"],                  True),
        ("DSR confidence",        exp_agg["dsr_confidence"],       sl_agg["dsr_confidence"],       True),
        ("Total trades",          float(exp_agg["total_trades"]),  float(sl_agg["total_trades"]),  True),
        ("Avg return / fold %",   exp_agg["avg_return_per_fold"], sl_agg["avg_return_per_fold"], True),
    ]
    for label, ev, sv, hb in rows:
        better = _pick(ev, sv, hb)
        print(f"  {label:<26}  {ev:>9.3f}  {sv:>9.3f}  {better:>10}")

    # Recommendation
    exp_sr = exp_agg["pooled_sharpe"]
    sl_sr  = sl_agg["pooled_sharpe"]
    if sl_sr > exp_sr + 0.05:
        rec    = "Sliding"
        reason = "Sliding Sharpe higher by > 0.05 — regime adaptation outweighs reduced data"
    else:
        rec    = "Expanding"
        reason = "Expanding Sharpe ≥ Sliding — more training data yields more stable estimates"

    print()
    print(f"  RECOMMENDATION: Use {rec} as primary.")
    print(f"  Rationale: {reason}")
    print("═" * W)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RSSS sliding walk-forward")
    parser.add_argument(
        "--folds", type=int, default=None,
        help="Limit number of folds to run (for smoke testing)"
    )
    args = parser.parse_args()

    t0 = time.time()

    SL_DIR.mkdir(parents=True, exist_ok=True)
    SL_FOLD_MODELS.mkdir(parents=True, exist_ok=True)

    # Redirect train_fold_models to save into the sliding fold-models dir
    _wfv.FOLD_MODELS = SL_FOLD_MODELS

    print("=" * 60)
    print("RSSS Walk-Forward Validation v2 — Sliding Window")
    print("=" * 60)
    print(f"  Train window:  {SLIDING_TRAIN_YEARS} years (fixed)")
    print(f"  Val window:    {SLIDING_VAL_MONTHS} months")
    print(f"  Test window:   {SLIDING_TEST_MONTHS} months")
    print(f"  Step:          {SLIDING_STEP_MONTHS} months")
    print(f"  Embargo:       {EMBARGO_DAYS} days")

    # ── Load features ─────────────────────────────────────────────────────────
    print("\nLoading features_v2.parquet...")
    df = pd.read_parquet(FEAT_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date"] >= WF_START) & (df["date"] <= WF_END)].copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    all_dates = sorted(df["date"].unique().tolist())
    print(f"  {len(df):,} rows  |  {len(all_dates)} trading days  "
          f"({all_dates[0]} → {all_dates[-1]})")

    # Drop tickers from locked architecture
    arch_path = BASE / "experiments" / "phase3_locked_architecture.json"
    try:
        with open(arch_path) as f:
            drop_tickers = set(json.load(f).get("drop_tickers", []))
    except FileNotFoundError:
        drop_tickers = set()
    print(f"  Drop tickers: {sorted(drop_tickers)}")

    # ── Download prices once ─────────────────────────────────────────────────
    sim_tickers = sorted(
        (set(df["ticker"].unique()) | {"SPY"}) - drop_tickers
    )
    print(f"\nDownloading prices for {len(sim_tickers)} tickers (2019-2026)...")

    raw = yf.download(
        sim_tickers, start=WF_START, end="2026-01-15",
        auto_adjust=True, progress=False, threads=True,
    )
    close_df = (raw["Close"]
                if isinstance(raw.columns, pd.MultiIndex)
                else raw[["Close"]].rename(columns={"Close": sim_tickers[0]}))

    price_lut: dict = {}
    ticker_series: dict = {}
    for t in sim_tickers:
        if t not in close_df.columns:
            continue
        s = close_df[t].dropna()
        ticker_series[t] = s
        for dt, p in s.items():
            price_lut[(t, dt.strftime("%Y-%m-%d"))] = float(p)

    for _, row in df.iterrows():
        k = (row["ticker"], row["date"])
        if k not in price_lut:
            price_lut[k] = float(row["close"])

    print(f"  Price lookup: {len(price_lut):,} entries")

    vol_lut: dict = {}
    for t, s in ticker_series.items():
        rv = s.pct_change().rolling(20, min_periods=10).std()
        for dt, v in rv.items():
            if pd.notna(v) and v > 0:
                vol_lut[(t, dt.strftime("%Y-%m-%d"))] = float(v)
    print(f"  Vol lookup:   {len(vol_lut):,} entries")

    spy_prices = {k[1]: v for k, v in price_lut.items() if k[0] == "SPY"}

    # ── Generate sliding folds ────────────────────────────────────────────────
    all_folds = generate_sliding_folds(all_dates)
    if args.folds is not None:
        all_folds = all_folds[: args.folds]

    print(f"\nSliding folds generated: {len(all_folds)}")
    for fold in all_folds[:3]:
        print(f"  Fold {fold['fold']:>2}: "
              f"train {fold['train_start']}→{fold['train_end']}  "
              f"test  {fold['test_start']}→{fold['test_end']}  "
              f"[{fold['regime']}]")
    if len(all_folds) > 3:
        print(f"  ... ({len(all_folds) - 3} more)")

    # ── Sliding walk-forward loop ─────────────────────────────────────────────
    fold_results: list[dict] = []
    print()

    for fold in all_folds:
        fn = fold["fold"]
        print(f"  Fold {fn:>2}/{len(all_folds)}  [{fold['regime']:<20}]  "
              f"test {fold['test_start']}→{fold['test_end']} ",
              end="", flush=True)

        # Sliding: train only within the fixed window
        train_df = df[
            (df["date"] >= fold["train_start"]) &
            (df["date"] <= fold["train_end"])
        ].copy()
        test_df  = df[
            (df["date"] >= fold["test_start"]) &
            (df["date"] <= fold["test_end"])
        ].copy()

        if len(train_df) < 200 or test_df.empty:
            print("  [SKIP: insufficient data]")
            continue

        models, is_ic = train_fold_models(train_df, fn)

        oos_tdays = sorted(test_df["date"].unique().tolist())
        oos = simulate(test_df, models, price_lut, vol_lut,
                       oos_tdays, spy_prices, drop_tickers)

        # IS backtest on last IS_LOOKBACK days of this fold's training window
        is_dates   = sorted(train_df["date"].unique().tolist())
        is_start   = is_dates[max(0, len(is_dates) - IS_LOOKBACK)]
        is_df      = train_df[train_df["date"] >= is_start].copy()
        is_tdays   = sorted(is_df["date"].unique().tolist())
        is_sim     = simulate(is_df, models, price_lut, vol_lut,
                              is_tdays, spy_prices, drop_tickers)

        print(f" OOS={oos['total_return_pct']:>+5.1f}%  "
              f"trades={oos['n_trades']:>3}  "
              f"SR={oos['sharpe']:>+5.2f}")

        fold_results.append({
            "fold":       fn,
            "regime":     fold["regime"],
            "train_start": fold["train_start"],
            "train_end":  fold["train_end"],
            "test_start": fold["test_start"],
            "test_end":   fold["test_end"],
            "oos":        oos,
            "is_sharpe":  is_sim["sharpe"],
            "is_ic":      is_ic,
        })

    if not fold_results:
        print("No folds completed — exiting.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\nAggregating {len(fold_results)} fold results...")
    agg = aggregate_results(fold_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results = []
    for fr in fold_results:
        rec = dict(fr)
        rec["oos"] = {k: v for k, v in fr["oos"].items()
                      if k not in ("daily_returns", "trades")}
        save_results.append(rec)

    out = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "wf_start":            WF_START,
        "wf_end":              WF_END,
        "sliding_train_years": SLIDING_TRAIN_YEARS,
        "sliding_val_months":  SLIDING_VAL_MONTHS,
        "sliding_test_months": SLIDING_TEST_MONTHS,
        "sliding_step_months": SLIDING_STEP_MONTHS,
        "purge_days":          PURGE_DAYS,
        "embargo_days":        EMBARGO_DAYS,
        "is_lookback":         IS_LOOKBACK,
        "feature_count":       len(FEATURE_COLS),
        "aggregate":           agg,
        "folds":               save_results,
    }

    with open(SL_RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved → {SL_RESULTS}")

    elapsed = round(time.time() - t0)
    print(f"Elapsed: {elapsed // 60}m {elapsed % 60}s")

    print_report(agg)

    # ── Compare with expanding ────────────────────────────────────────────────
    if EXP_RESULTS.exists():
        with open(EXP_RESULTS) as f:
            exp_data = json.load(f)
        _print_comparison(exp_data["aggregate"], agg)
    else:
        print(
            "\n  (No expanding-window results found at "
            f"{EXP_RESULTS} — run walk_forward_validation.py first for comparison)"
        )


if __name__ == "__main__":
    main()
