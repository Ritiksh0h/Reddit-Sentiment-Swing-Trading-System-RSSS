"""
Module: experiments/shared/backtest.py
Purpose: Shared backtest engine. Simulates daily ranking, position entry/exit,
         and equity curve construction. Identical rules across all experiments.
Phase: 2

Rules (locked — do not modify per experiment):
  - Entry at next-day open, simulated as close * (1 + slippage)
  - Exit at day 5 close
  - No shorting, no leverage
  - Max 3 concurrent positions
  - Trade only when predicted_return >= min_pred_return
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


def _last_valid_close(
    df: pd.DataFrame, ticker: str, before_date: pd.Timestamp
) -> Optional[float]:
    """Return the most recent valid close for ticker strictly before before_date."""
    rows = df[(df["ticker"] == ticker) & (df["date"] < before_date)].sort_values("date")
    if len(rows) == 0:
        return None
    val = float(rows["close"].iloc[-1])
    return val if val > 0 else None

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.thresholds import TAKE_PROFIT_CAP
from experiments.shared.metrics import compute_metrics

# Backtest constants — identical for all experiments (locked per spec)
STARTING_CAPITAL: float = 1_000.0
MAX_POSITIONS: int = 3
HOLD_DAYS: int = 5
SLIPPAGE: float = 0.001       # 0.1% per fill
FEE_PER_LEG: float = 0.0005  # 0.05% per leg
MIN_PRED_RETURN: float = 0.01  # 1% minimum predicted return to trade (halved from 0.02)


def run_backtest(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    pre_filter_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    starting_capital: float = STARTING_CAPITAL,
    max_positions: int = MAX_POSITIONS,
    hold_days: int = HOLD_DAYS,
    slippage: float = SLIPPAGE,
    fee_per_leg: float = FEE_PER_LEG,
    min_pred_return: float = MIN_PRED_RETURN,
    spy_return: float = 0.26,
) -> dict:
    """
    Simulate trading model predictions on test_df.

    Args:
        model: Fitted model with a .predict(X) method.
        test_df: Test set dataframe with 'date', 'ticker', 'close', feature_cols, target_return_5d.
        feature_cols: Features the model expects.
        pre_filter_fn: Optional callable applied to each day's candidate rows before ranking.
                       Used by Experiment A for the attention filter.
        starting_capital: Starting portfolio value.
        max_positions: Max concurrent open positions.
        hold_days: Trading days to hold each position.
        slippage: One-way slippage fraction.
        fee_per_leg: Commission per leg (entry + exit = 2x).
        min_pred_return: Minimum predicted return to open a position.
        spy_return: SPY total return over the test period (for alpha calculation).

    Returns:
        Dict with equity_curve, trades, and all metrics from compute_metrics().
    """
    df = test_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    X = df[feature_cols].fillna(0)
    df["pred_return"] = model.predict(X)

    trading_days = sorted(df["date"].unique())

    cash = starting_capital
    open_positions: list[dict] = []
    equity_curve: list[float] = [starting_capital]
    trades: list[dict] = []
    # Fix 1: track when each ticker last closed to enforce per-ticker cooldown
    recently_closed: dict[str, pd.Timestamp] = {}
    cooldown_window = pd.Timedelta(days=hold_days * 1.4)

    for day in trading_days:
        today_df = df[df["date"] == day]

        # ----------------------------------------------------------------
        # 1. Close positions that have reached hold_days
        # ----------------------------------------------------------------
        still_open = []
        for pos in open_positions:
            days_held = (day - pos["entry_date"]).days
            if days_held >= hold_days * 1.4:  # ~7 calendar days ≈ 5 trading days
                ticker_rows = today_df[today_df["ticker"] == pos["ticker"]]

                # Fix 2: use last valid close if today's price is missing
                if len(ticker_rows) > 0:
                    exit_close = float(ticker_rows["close"].iloc[0])
                else:
                    exit_close = _last_valid_close(df, pos["ticker"], day)
                    if exit_close is None:
                        still_open.append(pos)  # no price at all — hold another day
                        continue

                exit_price = exit_close * (1 - slippage)
                gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                fee = exit_price * pos["shares"] * fee_per_leg
                net_pnl = gross_pnl - fee

                cash += pos["entry_price"] * pos["shares"] + net_pnl

                # Fix 1: record close date for cooldown enforcement
                recently_closed[pos["ticker"]] = day

                trades.append(
                    {
                        "ticker": pos["ticker"],
                        "entry_date": str(pos["entry_date"].date()),
                        "exit_date": str(day.date()),
                        "entry_price": round(pos["entry_price"], 4),
                        "exit_price": round(exit_price, 4),
                        "shares": pos["shares"],
                        "pred_return": round(pos["pred_return"], 6),
                        "gross_pnl": round(gross_pnl, 4),
                        "pnl": round(net_pnl, 4),
                        "exit_reason": "hold_days",
                    }
                )
            else:
                still_open.append(pos)

        open_positions = still_open

        # ----------------------------------------------------------------
        # 2. Open new positions if capacity allows
        # ----------------------------------------------------------------
        n_open = len(open_positions)
        slots = max_positions - n_open

        if slots > 0 and len(today_df) > 0:
            candidates = today_df.copy()

            # Apply optional pre-filter (Experiment A attention filter)
            if pre_filter_fn is not None:
                candidates = pre_filter_fn(candidates)

            # Filter by minimum predicted return
            candidates = candidates[candidates["pred_return"] >= min_pred_return]

            # Fix 1: exclude tickers currently held or within cooldown window
            open_tickers = {p["ticker"] for p in open_positions}
            in_cooldown = candidates["ticker"].apply(
                lambda t: bool(
                    t in recently_closed
                    and (day - recently_closed[t]) < cooldown_window
                )
            ).astype(bool)
            candidates = candidates[
                ~candidates["ticker"].isin(open_tickers) & ~in_cooldown
            ]

            # Rank by predicted return, take top-N
            candidates = candidates.nlargest(slots, "pred_return")

            for _, row in candidates.iterrows():
                if cash <= 0:
                    break

                # Equal-weight position sizing across max_positions slots
                position_value = starting_capital / max_positions
                position_value = min(position_value, cash)

                entry_price = float(row["close"]) * (1 + slippage)
                fee_entry = entry_price * (position_value / entry_price) * fee_per_leg
                # Reduce position value by entry fee
                shares = (position_value - fee_entry) / entry_price

                if shares <= 0:
                    continue

                cash -= shares * entry_price + fee_entry

                open_positions.append(
                    {
                        "ticker": row["ticker"],
                        "entry_date": day,
                        "entry_price": entry_price,
                        "shares": shares,
                        "pred_return": float(row["pred_return"]),
                    }
                )

        # ----------------------------------------------------------------
        # 3. Take-profit check + mark-to-market equity curve
        # ----------------------------------------------------------------
        still_open_after_tp = []
        open_value = 0.0

        for pos in open_positions:
            ticker_rows = today_df[today_df["ticker"] == pos["ticker"]]
            if len(ticker_rows) > 0:
                mtm_price = float(ticker_rows["close"].iloc[0])
            else:
                mtm_price = pos["entry_price"]  # hold at cost if no quote

            unrealized_return = (mtm_price / pos["entry_price"]) - 1

            if unrealized_return >= TAKE_PROFIT_CAP:
                # Close at today's close (minus slippage)
                exit_price = mtm_price * (1 - slippage)
                gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                fee = exit_price * pos["shares"] * fee_per_leg
                net_pnl = gross_pnl - fee
                cash += pos["entry_price"] * pos["shares"] + net_pnl
                recently_closed[pos["ticker"]] = day
                trades.append(
                    {
                        "ticker": pos["ticker"],
                        "entry_date": str(pos["entry_date"].date()),
                        "exit_date": str(day.date()),
                        "entry_price": round(pos["entry_price"], 4),
                        "exit_price": round(exit_price, 4),
                        "shares": pos["shares"],
                        "pred_return": round(pos["pred_return"], 6),
                        "gross_pnl": round(gross_pnl, 4),
                        "pnl": round(net_pnl, 4),
                        "exit_reason": "take_profit_cap",
                    }
                )
                # Freed cash goes into equity curve immediately
                open_value += 0.0
            else:
                still_open_after_tp.append(pos)
                open_value += pos["shares"] * mtm_price

        open_positions = still_open_after_tp
        equity_curve.append(cash + open_value)

    # Force-close any remaining positions at last available price
    last_day = trading_days[-1]
    last_df = df[df["date"] == last_day]
    for pos in open_positions:
        ticker_rows = last_df[last_df["ticker"] == pos["ticker"]]

        # Fix 2: use last valid close; skip trade entirely if no price exists
        if len(ticker_rows) > 0:
            exit_close = float(ticker_rows["close"].iloc[0])
        else:
            exit_close = _last_valid_close(
                df, pos["ticker"], last_day + pd.Timedelta(days=1)
            )
            if exit_close is None:
                continue  # no price ever found — don't record a zero-return trade

        exit_price = exit_close * (1 - slippage)
        gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        fee = exit_price * pos["shares"] * fee_per_leg
        net_pnl = gross_pnl - fee

        trades.append(
            {
                "ticker": pos["ticker"],
                "entry_date": str(pos["entry_date"].date()),
                "exit_date": str(last_day.date()),
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(exit_price, 4),
                "shares": pos["shares"],
                "pred_return": round(pos["pred_return"], 6),
                "gross_pnl": round(gross_pnl, 4),
                "pnl": round(net_pnl, 4),
            }
        )

    metrics = compute_metrics(equity_curve, trades, spy_return)
    metrics["equity_curve"] = [round(v, 4) for v in equity_curve]
    metrics["trades"] = trades
    return metrics
