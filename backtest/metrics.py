"""
Module: backtest/metrics.py
Purpose: Compute all required backtest performance metrics (§10.3).
Phase: 5 — Backtesting Engine
Dependencies: numpy, pandas, utils/logger.py
Last modified: 2026-06-10
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR: int = 252


def compute_metrics(
    equity_curve: list[float] | pd.Series,
    trades: list[dict],
    spy_annualized_return: Optional[float] = None,
) -> dict:
    """
    Compute all §10.3 required backtest metrics from an equity curve and trade list.

    Args:
        equity_curve: Daily portfolio values (list or Series), indexed chronologically
        trades: List of trade dicts, each with keys:
                entry_date, exit_date, pnl (float), entry_price, exit_price
        spy_annualized_return: SPY benchmark annualized return for alpha calculation.
                               None if benchmark not available.

    Returns:
        Dict of metrics per §10.3.
    """
    equity = np.array(equity_curve, dtype=float)

    if len(equity) < 2:
        log.warning("metrics_insufficient_equity_curve", length=len(equity))
        return {}

    # Daily returns
    daily_returns = np.diff(equity) / equity[:-1]
    n_days = len(daily_returns)

    # Total and annualized return
    total_return = (equity[-1] / equity[0]) - 1.0
    annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0

    # Sharpe ratio
    ret_std = float(np.std(daily_returns, ddof=1))
    sharpe_ratio = (
        float(np.mean(daily_returns)) / ret_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if ret_std > 0
        else 0.0
    )

    # Sortino ratio (downside std only)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = float(np.std(downside_returns, ddof=1)) if len(downside_returns) > 1 else 0.0
    sortino_ratio = (
        float(np.mean(daily_returns)) / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_std > 0
        else 0.0
    )

    # Maximum drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    max_drawdown = float(np.min(drawdowns))

    # Trade-level metrics
    if trades:
        pnls = [t.get("pnl", 0.0) for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        win_rate = len(winning) / len(pnls)
        profit_factor = (
            sum(winning) / abs(sum(losing))
            if losing and sum(losing) != 0
            else float("inf")
        )

        holding_periods = []
        for t in trades:
            entry = pd.Timestamp(t.get("entry_date", ""))
            exit_ = pd.Timestamp(t.get("exit_date", ""))
            if pd.notna(entry) and pd.notna(exit_):
                holding_periods.append((exit_ - entry).days)
        avg_holding_days = float(np.mean(holding_periods)) if holding_periods else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_holding_days = 0.0

    # Alpha vs benchmark
    benchmark_alpha = (
        annualized_return - spy_annualized_return
        if spy_annualized_return is not None
        else None
    )

    metrics = {
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        "total_return": round(total_return, 4),
        "annualized_return": round(annualized_return, 4),
        "avg_holding_days": round(avg_holding_days, 2),
        "benchmark_alpha": round(benchmark_alpha, 4) if benchmark_alpha is not None else None,
        "n_trades": len(trades),
        "n_trading_days": n_days,
    }

    log.info("metrics_computed", **{k: v for k, v in metrics.items() if v is not None})
    return metrics
