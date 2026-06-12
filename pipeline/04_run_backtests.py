#!/usr/bin/env python3
"""
Module: pipeline/04_run_backtests.py
Purpose: Simulate trading from the Combined model predictions on the 2024 test set.
         Compute full §10.3 performance metrics. Compare vs SPY buy-and-hold.

Phase: 1 — Research Pipeline
Input:  data/features/features.parquet
        models/registry/combined/ (from Script 03)
Output: reports/backtest_report.json, reports/backtest_report.html

Usage:
    python pipeline/04_run_backtests.py
    python pipeline/04_run_backtests.py --debug
Last modified: 2026-06-11
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import FEATURES_PARQUET, MODELS_DIR, REPORTS_DIR
from config.thresholds import (
    STARTING_CAPITAL,
    MAX_POSITIONS,
    HOLD_DAYS,
    SLIPPAGE,
    FEE_PER_LEG,
    MIN_PRED_RETURN,
    MIN_CONFIDENCE,
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
# Load model
# ---------------------------------------------------------------------------

def load_model(model_name: str = "combined"):
    """Load trained model from registry."""
    model_dir = MODELS_DIR / model_name
    pkl_path = model_dir / "model.pkl"
    meta_path = model_dir / "metadata.json"

    if not pkl_path.exists():
        log.error("model_not_found", path=str(pkl_path))
        return None, None

    model = joblib.load(pkl_path)
    with open(meta_path) as f:
        meta = json.load(f)

    log.info("model_loaded", name=model_name, features=len(meta["feature_cols"]))
    return model, meta


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

def run_ml_backtest(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    strategy_name: str = "combined_model",
    capital: float = STARTING_CAPITAL,
    max_pos: int = MAX_POSITIONS,
    hold_days: int = HOLD_DAYS,
    slippage: float = SLIPPAGE,
    fee_per_leg: float = FEE_PER_LEG,
    min_pred_return: float = MIN_PRED_RETURN,
) -> dict:
    """
    Simulate ML-signal-driven trading on test set.

    Signal: enter when model predicts return > min_pred_return.
    Rank: highest predicted return first.
    Hold: hold_days calendar trading days (same as baseline).

    Args:
        test_df: Test feature DataFrame (reset index, sorted by date)
        preds: Model predictions aligned to test_df rows
        strategy_name: Name for logging/reporting
        capital: Starting capital
        max_pos: Max concurrent positions
        hold_days: Hold period
        slippage: Entry/exit slippage fraction
        fee_per_leg: Commission per leg
        min_pred_return: Minimum predicted return to enter a position

    Returns:
        Performance metrics dict.
    """
    df = test_df.copy().reset_index(drop=True)
    df["pred"] = preds

    trading_days = sorted(df["date"].unique())
    close_map: dict[tuple[str, str], float] = {
        (r["ticker"], r["date"]): r["close"] for _, r in df.iterrows()
    }

    positions: list[dict] = []
    trades: list[dict] = []
    equity_curve: list[float] = [capital]
    cash = capital

    for i, day in enumerate(trading_days):
        day_str = str(day)
        day_df = df[df["date"] == day_str].copy()

        # Exit positions due for close
        still_open = []
        for pos in positions:
            age = i - pos["entry_day_idx"]
            if age >= hold_days:
                exit_price = close_map.get((pos["ticker"], day_str))
                if exit_price is not None:
                    proceeds = pos["shares"] * exit_price * (1 - slippage)
                    fee = proceeds * fee_per_leg
                    pnl = proceeds - fee - pos["cost"]
                    cash += proceeds - fee
                    trades.append({
                        "ticker": pos["ticker"],
                        "entry_date": pos["entry_date"],
                        "exit_date": day_str,
                        "pnl": pnl,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "pred_return": pos["pred_return"],
                        "actual_return": float(exit_price / pos["entry_price"] - 1),
                    })
                else:
                    still_open.append(pos)
                    continue
            else:
                still_open.append(pos)
        positions = still_open

        # Open new positions on signal
        slots = max_pos - len(positions)
        if slots > 0 and not day_df.empty:
            candidates = (
                day_df[day_df["pred"] >= min_pred_return]
                .nlargest(slots, "pred")
            )
            for _, row in candidates.iterrows():
                entry_price = close_map.get((row["ticker"], day_str))
                if entry_price is None or cash <= 0:
                    continue
                alloc = cash / max(slots, 1)
                alloc = min(alloc, cash * 0.5)
                buy_price = entry_price * (1 + slippage)
                fee = alloc * fee_per_leg
                cost = alloc + fee
                if cost > cash:
                    continue
                shares = alloc / buy_price
                cash -= cost
                positions.append({
                    "ticker": row["ticker"],
                    "entry_date": day_str,
                    "entry_day_idx": i,
                    "entry_price": buy_price,
                    "shares": shares,
                    "cost": cost,
                    "pred_return": float(row["pred"]),
                })

        # Mark-to-market
        open_val = sum(
            pos["shares"] * close_map.get((pos["ticker"], day_str), pos["entry_price"])
            for pos in positions
        )
        equity_curve.append(cash + open_val)

    # Close remaining
    for pos in positions:
        last_price = close_map.get((pos["ticker"], str(trading_days[-1])), pos["entry_price"])
        proceeds = pos["shares"] * last_price * (1 - slippage)
        fee = proceeds * fee_per_leg
        pnl = proceeds - fee - pos["cost"]
        cash += proceeds - fee
        trades.append({
            "ticker": pos["ticker"],
            "entry_date": pos["entry_date"],
            "exit_date": str(trading_days[-1]),
            "pnl": pnl,
            "entry_price": pos["entry_price"],
            "exit_price": last_price,
            "pred_return": pos["pred_return"],
            "actual_return": float(last_price / pos["entry_price"] - 1),
        })

    # Metrics
    eq = np.array(equity_curve, dtype=float)
    daily_rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
    daily_rets = daily_rets[np.isfinite(daily_rets)]

    total_ret = (cash / capital) - 1.0
    n_days = max(len(daily_rets), 1)
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1

    ret_std = float(np.std(daily_rets)) if len(daily_rets) > 1 else 1e-9
    sharpe = float(np.mean(daily_rets)) / ret_std * np.sqrt(252) if ret_std > 0 else 0.0

    running_max = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / running_max - 1))

    pnls = [t["pnl"] for t in trades]
    win_rate = sum(1 for p in pnls if p > 0) / max(len(pnls), 1)

    # IC on trades (predicted vs actual)
    pred_rets = [t["pred_return"] for t in trades]
    actual_rets = [t["actual_return"] for t in trades]
    trade_ic = 0.0
    if len(pred_rets) >= 5:
        corr, _ = spearmanr(pred_rets, actual_rets)
        trade_ic = float(corr) if np.isfinite(corr) else 0.0

    log.info(
        "ml_backtest_complete",
        strategy=strategy_name,
        total_return=round(total_ret, 4),
        sharpe=round(sharpe, 3),
        n_trades=len(trades),
        win_rate=round(win_rate, 3),
        trade_ic=round(trade_ic, 4),
    )

    return {
        "strategy": strategy_name,
        "total_return": round(total_ret, 4),
        "annualized_return": round(ann_ret, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "n_trades": len(trades),
        "trade_ic": round(trade_ic, 4),
        "equity_curve": [round(v, 2) for v in eq.tolist()],
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------

def get_spy_benchmark(start: str, end: str) -> float:
    try:
        spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if spy.empty:
            return 0.0
        # Handle multi-level columns (yfinance MultiIndex)
        if isinstance(spy.columns, pd.MultiIndex):
            close = spy["Close"].squeeze()
        else:
            close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
        close = close.dropna()
        return float(close.iloc[-1]) / float(close.iloc[0]) - 1.0
    except Exception as e:
        log.error("spy_failed", error=str(e))
        return 0.0


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html_report(result: dict, spy_return: float) -> str:
    r = result
    beats = "✅ BEATS SPY" if r["total_return"] > spy_return else "❌ UNDERPERFORMS SPY"
    high_sharpe_warn = ""
    if r["sharpe_ratio"] > 2.5:
        high_sharpe_warn = (
            "<p style='color:red;'><b>⚠️ WARNING: Sharpe > 2.5 — verify no data leakage</b></p>"
        )
    if r["win_rate"] > 0.65:
        high_sharpe_warn += (
            "<p style='color:red;'><b>⚠️ WARNING: Win rate > 65% — verify no data leakage</b></p>"
        )

    return f"""<!DOCTYPE html>
<html><head><title>RSSS Backtest Report</title>
<style>
  body {{ font-family: monospace; padding: 20px; }}
  table {{ border-collapse: collapse; width: 60%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; }}
  th {{ background: #f0f0f0; text-align: left; }}
</style></head><body>
<h2>RSSS Phase 1 — ML Backtest (2024 Test Set)</h2>
{high_sharpe_warn}
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Return</td><td>{r['total_return']:.1%}</td></tr>
  <tr><td>Annualized Return</td><td>{r['annualized_return']:.1%}</td></tr>
  <tr><td>Sharpe Ratio</td><td>{r['sharpe_ratio']:.2f}</td></tr>
  <tr><td>Max Drawdown</td><td>{r['max_drawdown']:.1%}</td></tr>
  <tr><td>Win Rate</td><td>{r['win_rate']:.1%}</td></tr>
  <tr><td>N Trades</td><td>{r['n_trades']}</td></tr>
  <tr><td>Trade IC</td><td>{r['trade_ic']:.4f}</td></tr>
  <tr><td>SPY Benchmark</td><td>{spy_return:.1%}</td></tr>
  <tr><td>Alpha (vs SPY)</td><td>{r['total_return'] - spy_return:.1%}</td></tr>
  <tr><th>Verdict</th><th>{beats}</th></tr>
</table>
<p><small>Generated by pipeline/04_run_backtests.py</small></p>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — ML Backtests")
    parser.add_argument("--debug", action="store_true",
                        help="Run on 2 tickers only")
    parser.add_argument("--model", default="combined",
                        help="Model to backtest (market/reddit/combined)")
    args = parser.parse_args()

    # Load model
    model, meta = load_model(args.model)
    if model is None:
        print(f"ERROR: Model '{args.model}' not found. Run 03_train_models.py first.")
        sys.exit(1)

    # Load features
    if not FEATURES_PARQUET.exists():
        print(f"ERROR: {FEATURES_PARQUET} not found.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PARQUET)
    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        print("ERROR: No test data.")
        sys.exit(1)

    if args.debug:
        tickers = test_df["ticker"].unique()[:2]
        test_df = test_df[test_df["ticker"].isin(tickers)]
        log.info("debug_mode", tickers=list(tickers))

    # Fill NaN for features (same logic as training)
    feat_cols = meta["feature_cols"]
    missing_cols = [c for c in feat_cols if c not in test_df.columns]
    if missing_cols:
        log.warning("feature_cols_missing", cols=missing_cols)
        feat_cols = [c for c in feat_cols if c in test_df.columns]

    X_test = test_df[feat_cols].copy()

    # Determine fill strategy based on model type
    if args.model == "reddit":
        X_test = X_test.fillna(0)
    else:
        # Use column medians computed from column (approximation — ideally saved with model)
        X_test = X_test.fillna(X_test.median())

    preds = model.predict(X_test)
    log.info("predictions_generated", n=len(preds), mean=round(float(np.mean(preds)), 4))

    # Run backtest
    start_date = str(test_df["date"].min())
    end_date = str(test_df["date"].max())
    spy_return = get_spy_benchmark(start_date, end_date)

    result = run_ml_backtest(test_df, preds, strategy_name=f"{args.model}_model")
    result["spy_return"] = spy_return
    result["alpha"] = round(result["total_return"] - spy_return, 4)
    result["beats_spy"] = result["total_return"] > spy_return

    # Leakage warnings (per spec: Sharpe > 2.5 or win > 65% → assume leakage first)
    leakage_flags = []
    if result["sharpe_ratio"] > 2.5:
        leakage_flags.append(f"Sharpe={result['sharpe_ratio']:.2f} > 2.5")
    if result["win_rate"] > 0.65:
        leakage_flags.append(f"win_rate={result['win_rate']:.1%} > 65%")
    result["leakage_flags"] = leakage_flags

    # Save report
    report = {
        "run_date": pd.Timestamp.now().isoformat(),
        "model": args.model,
        "test_period": f"{start_date} → {end_date}",
        **{k: v for k, v in result.items() if k not in ("equity_curve", "trades")},
        "equity_curve": result["equity_curve"],
    }

    report_json = REPORTS_DIR / "backtest_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("backtest_report_saved", path=str(report_json))

    html = generate_html_report(result, spy_return)
    (REPORTS_DIR / "backtest_report.html").write_text(html)

    print("\n=== ML BACKTEST RESULTS (2024) ===")
    print(f"  Model:          {args.model}")
    print(f"  Total Return:   {result['total_return']:+.1%}")
    print(f"  Sharpe:         {result['sharpe_ratio']:.2f}")
    print(f"  Win Rate:       {result['win_rate']:.0%}")
    print(f"  Max Drawdown:   {result['max_drawdown']:.1%}")
    print(f"  Trade IC:       {result['trade_ic']:.4f}")
    print(f"  SPY Benchmark:  {spy_return:+.1%}")
    print(f"  Alpha:          {result['alpha']:+.1%}")
    if leakage_flags:
        print(f"\n  ⚠️  LEAKAGE FLAGS: {', '.join(leakage_flags)}")
        print("     Numbers too good — check for data leakage before proceeding")
    else:
        print("\n  ✅ Results within expected range")
    print(f"  Report saved to {REPORTS_DIR}/")
    print("=" * 40)


if __name__ == "__main__":
    main()
