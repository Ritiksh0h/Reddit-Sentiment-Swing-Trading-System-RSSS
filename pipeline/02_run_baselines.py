#!/usr/bin/env python3
"""
Module: pipeline/02_run_baselines.py
Purpose: Test 3 rule-based strategies on 2024 test data before any ML.
         If these can't beat random selection, stop and reassess.

         Strategy A — Top Mention Growth
         Strategy B — Top Sentiment
         Strategy C — Attention + Volume

Phase: 1 — Research Pipeline
Input:  data/features/features.parquet
Output: reports/baseline_report.json, reports/baseline_report.html

Usage:
    python pipeline/02_run_baselines.py
    python pipeline/02_run_baselines.py --debug
Last modified: 2026-06-11
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import FEATURES_PARQUET, REPORTS_DIR
from config.thresholds import (
    STARTING_CAPITAL,
    MAX_POSITIONS,
    HOLD_DAYS,
    SLIPPAGE,
    FEE_PER_LEG,
    MIN_POST_COUNT,
)
from utils.logger import get_logger

log = get_logger(__name__)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Backtest Engine (shared by all 3 strategies)
# ---------------------------------------------------------------------------

def run_strategy_backtest(
    test_df: pd.DataFrame,
    select_fn,
    strategy_name: str,
    capital: float = STARTING_CAPITAL,
    max_pos: int = MAX_POSITIONS,
    hold_days: int = HOLD_DAYS,
    slippage: float = SLIPPAGE,
    fee_per_leg: float = FEE_PER_LEG,
) -> dict:
    """
    Simulate a rule-based strategy on the test set.

    Mechanics:
    - Entry at next-day open (approximated as close * 1.001 slippage)
    - Exit at hold_days close
    - Equal weight across max_pos positions
    - No shorting, no leverage

    Args:
        test_df: Feature DataFrame filtered to test split
        select_fn: Callable(day_df) → list of tickers to buy (max_pos items)
        strategy_name: For logging
        capital: Starting capital
        max_pos: Max concurrent positions
        hold_days: Hold period in trading days
        slippage: One-way slippage fraction
        fee_per_leg: Commission per leg

    Returns:
        Performance metrics dict.
    """
    trading_days = sorted(test_df["date"].unique())
    log.info("backtest_start", strategy=strategy_name, days=len(trading_days))

    # Build close price lookup: {(ticker, date_str) → close}
    close_map: dict[tuple[str, str], float] = {}
    for _, row in test_df.iterrows():
        close_map[(row["ticker"], row["date"])] = row["close"]

    # Position tracking: {ticker: {entry_day_idx, entry_price, exit_day_idx, cost}}
    positions: list[dict] = []  # active positions
    trades: list[dict] = []
    equity_curve: list[float] = [capital]
    cash = capital

    for i, day in enumerate(trading_days):
        day_str = str(day)
        day_df = test_df[test_df["date"] == day_str]

        # Exit positions that have held >= hold_days
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
                    })
                else:
                    # No price data — carry position
                    still_open.append(pos)
                    continue
            else:
                still_open.append(pos)
        positions = still_open

        # Open new positions if we have capacity
        n_open = len(positions)
        slots = max_pos - n_open
        if slots > 0 and not day_df.empty:
            candidates = select_fn(day_df)[:slots]
            for ticker in candidates:
                entry_price = close_map.get((ticker, day_str))
                if entry_price is None or cash <= 0:
                    continue
                alloc = (cash / slots) if slots > 0 else 0
                alloc = min(alloc, cash * 0.5)  # don't deploy all cash at once
                buy_price = entry_price * (1 + slippage)
                fee = alloc * fee_per_leg
                cost = alloc + fee
                if cost > cash:
                    continue
                shares = alloc / buy_price
                cash -= cost
                positions.append({
                    "ticker": ticker,
                    "entry_date": day_str,
                    "entry_day_idx": i,
                    "entry_price": buy_price,
                    "shares": shares,
                    "cost": cost,
                })

        # Mark-to-market portfolio value
        open_value = sum(
            pos["shares"] * close_map.get((pos["ticker"], day_str), pos["entry_price"])
            for pos in positions
        )
        equity_curve.append(cash + open_value)

    # Close remaining positions at last price
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
        })

    final_equity = cash
    eq = np.array(equity_curve, dtype=float)
    daily_returns = np.diff(eq) / eq[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    total_return = (final_equity / capital) - 1.0
    n_days = len(daily_returns)
    ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0.0

    ret_std = float(np.std(daily_returns)) if len(daily_returns) > 1 else 1e-9
    sharpe = float(np.mean(daily_returns)) / ret_std * np.sqrt(252) if ret_std > 0 else 0.0

    running_max = np.maximum.accumulate(eq)
    max_dd = float(np.min(eq / running_max - 1))

    pnls = [t["pnl"] for t in trades]
    win_rate = sum(1 for p in pnls if p > 0) / max(len(pnls), 1)

    log.info(
        "backtest_complete",
        strategy=strategy_name,
        total_return=round(total_return, 4),
        sharpe=round(sharpe, 3),
        n_trades=len(trades),
        win_rate=round(win_rate, 3),
    )

    return {
        "strategy": strategy_name,
        "total_return": round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "n_trades": len(trades),
        "equity_curve": [round(v, 2) for v in eq.tolist()],
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# Strategy Selectors
# ---------------------------------------------------------------------------

def strategy_a_select(day_df: pd.DataFrame) -> list[str]:
    """Strategy A: Top mention_growth_1d, min post_count_1d >= 3."""
    filtered = day_df[day_df["post_count_1d"] >= MIN_POST_COUNT].copy()
    ranked = filtered.nlargest(MAX_POSITIONS, "mention_growth_1d")
    return ranked["ticker"].tolist()


def strategy_b_select(day_df: pd.DataFrame) -> list[str]:
    """Strategy B: Top weighted_sentiment, positive avg_sentiment only."""
    filtered = day_df[
        (day_df["post_count_1d"] >= MIN_POST_COUNT)
        & (day_df["avg_sentiment_1d"] > 0)
    ].copy()
    ranked = filtered.nlargest(MAX_POSITIONS, "weighted_sentiment")
    return ranked["ticker"].tolist()


def strategy_c_select(day_df: pd.DataFrame) -> list[str]:
    """Strategy C: Attention + Volume filter, ranked by total_upvotes_1d."""
    filtered = day_df[
        (day_df["post_count_1d"] >= 5)
        & (day_df["relative_volume"] >= 1.5)
        & (day_df["mention_growth_1d"] >= 0.3)
    ].copy()
    ranked = filtered.nlargest(MAX_POSITIONS, "total_upvotes_1d")
    return ranked["ticker"].tolist()


# ---------------------------------------------------------------------------
# SPY Benchmark
# ---------------------------------------------------------------------------

def get_spy_return(start_date: str, end_date: str) -> float:
    """Compute SPY buy-and-hold return over the period."""
    try:
        spy = yf.download("SPY", start=start_date, end=end_date,
                          auto_adjust=True, progress=False)
        if spy.empty:
            log.warning("spy_data_empty")
            return 0.0
        # Handle multi-level columns (yfinance >= 0.2 returns (metric, ticker) MultiIndex)
        if isinstance(spy.columns, pd.MultiIndex):
            close = spy["Close"].squeeze()
        else:
            close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
        close = close.dropna()
        ret = float(close.iloc[-1]) / float(close.iloc[0]) - 1.0
        log.info("spy_return_computed", ret=round(ret, 4))
        return ret
    except Exception as e:
        log.error("spy_return_failed", error=str(e))
        return 0.0


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html_report(results: list[dict], spy_return: float) -> str:
    """Generate a simple HTML summary of baseline strategy results."""
    rows = ""
    for r in results:
        beats = "✅" if r["total_return"] > spy_return else "❌"
        rows += f"""
        <tr>
            <td>{r['strategy']}</td>
            <td>{r['total_return']:.1%}</td>
            <td>{r['annualized_return']:.1%}</td>
            <td>{r['sharpe_ratio']:.2f}</td>
            <td>{r['max_drawdown']:.1%}</td>
            <td>{r['win_rate']:.1%}</td>
            <td>{r['n_trades']}</td>
            <td>{beats}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><title>RSSS Baseline Strategies</title>
<style>
  body {{ font-family: monospace; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td:first-child {{ text-align: left; }}
</style></head><body>
<h2>RSSS Phase 1 — Baseline Strategies (2024 Test Set)</h2>
<table>
  <tr>
    <th>Strategy</th><th>Total Return</th><th>Ann. Return</th>
    <th>Sharpe</th><th>Max DD</th><th>Win Rate</th>
    <th>N Trades</th><th>Beats SPY?</th>
  </tr>
  {rows}
  <tr style="background:#e8f4e8">
    <td><b>SPY Benchmark</b></td>
    <td colspan="6"><b>{spy_return:.1%}</b></td>
    <td>—</td>
  </tr>
</table>
<p><small>Generated by pipeline/02_run_baselines.py</small></p>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Phase 1 — Baseline Strategies")
    parser.add_argument("--debug", action="store_true",
                        help="Run on first 50 test-set rows")
    args = parser.parse_args()

    if not FEATURES_PARQUET.exists():
        log.error("features_not_found", path=str(FEATURES_PARQUET))
        print(f"ERROR: {FEATURES_PARQUET} not found. Run 01_feature_builder.py first.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PARQUET)
    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        log.error("no_test_data")
        print("ERROR: No test data (2024) in features. Check train/test split.")
        sys.exit(1)

    if args.debug:
        tickers_sample = test_df["ticker"].unique()[:5]
        test_df = test_df[test_df["ticker"].isin(tickers_sample)]
        log.info("debug_mode", rows=len(test_df), tickers=list(tickers_sample))

    log.info(
        "baseline_start",
        test_rows=len(test_df),
        test_dates=f"{test_df['date'].min()} → {test_df['date'].max()}",
        tickers=test_df["ticker"].nunique(),
    )

    start_date = str(test_df["date"].min())
    end_date = str(test_df["date"].max())
    spy_return = get_spy_return(start_date, end_date)

    strategies = [
        ("mention_growth", strategy_a_select),
        ("top_sentiment",  strategy_b_select),
        ("attention_volume", strategy_c_select),
    ]

    results = []
    for name, select_fn in strategies:
        result = run_strategy_backtest(test_df, select_fn, strategy_name=name)
        result["spy_return"] = spy_return
        result["alpha"] = round(result["total_return"] - spy_return, 4)
        result["beats_spy"] = result["total_return"] > spy_return
        result["verdict"] = (
            "BEATS_SPY" if result["beats_spy"] else "UNDERPERFORMS_SPY"
        )
        results.append(result)

    # --- Save results ---
    # Strip equity_curve and trades from JSON (too large; keep in separate key)
    summary_results = []
    for r in results:
        s = {k: v for k, v in r.items() if k not in ("equity_curve", "trades")}
        summary_results.append(s)

    report = {
        "run_date": pd.Timestamp.now().isoformat(),
        "test_period": f"{start_date} → {end_date}",
        "spy_return": spy_return,
        "strategies": summary_results,
    }

    report_json = REPORTS_DIR / "baseline_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("baseline_report_saved", path=str(report_json))

    html = generate_html_report(results, spy_return)
    report_html = REPORTS_DIR / "baseline_report.html"
    report_html.write_text(html)
    log.info("baseline_html_saved", path=str(report_html))

    # --- Console summary ---
    print("\n=== BASELINE STRATEGY RESULTS (2024) ===")
    print(f"  SPY benchmark: {spy_return:.1%}")
    print()
    for r in results:
        beats = "✓ beats SPY" if r["beats_spy"] else "✗ underperforms SPY"
        print(f"  {r['strategy']:20s}  return={r['total_return']:+.1%}  "
              f"sharpe={r['sharpe_ratio']:.2f}  win={r['win_rate']:.0%}  {beats}")
    print()

    # Decision gate per spec
    any_beats = any(r["beats_spy"] for r in results)
    strat_c = next((r for r in results if r["strategy"] == "attention_volume"), None)
    if strat_c and strat_c["win_rate"] < 0.50:
        print("  ⚠️  GATE: Strategy C win rate < 50% — review findings before Script 03")
    print(f"  Reports saved to {REPORTS_DIR}/")
    print("=" * 45)


if __name__ == "__main__":
    main()
