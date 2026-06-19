"""
Module: experiments/shared/metrics.py
Purpose: Compute standard backtest performance metrics. Used by all three experiments.
Phase: 2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_metrics(
    equity_curve: list[float],
    trades: list[dict],
    spy_return: float,
) -> dict:
    """
    Compute all standard backtest metrics from an equity curve and trade list.

    Args:
        equity_curve: Portfolio value at each step (daily or per-trade).
        trades: List of dicts, each with at least a 'pnl' key.
        spy_return: SPY total return over the same period (e.g. 0.26 for 26%).

    Returns:
        Dict of performance metrics.
    """
    if len(equity_curve) < 2:
        return _empty_metrics(spy_return)

    curve = pd.Series(equity_curve, dtype=float)
    returns = curve.pct_change().dropna()

    total_return = (curve.iloc[-1] / curve.iloc[0]) - 1
    n_periods = len(returns)

    annualized_return = (
        ((curve.iloc[-1] / curve.iloc[0]) ** (252.0 / n_periods)) - 1
        if n_periods > 0
        else 0.0
    )

    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    downside = returns[returns < 0].std()
    sortino = float(returns.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0

    max_drawdown = float((curve / curve.cummax() - 1).min())

    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
    win_rate = len(winning_trades) / max(len(trades), 1)

    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    beats_spy = bool(total_return > spy_return)

    return {
        "total_return": round(float(total_return), 6),
        "annualized_return": round(float(annualized_return), 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 6),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(float(profit_factor), 4),
        "n_trades": len(trades),
        "spy_return": round(float(spy_return), 6),
        "alpha": round(float(total_return - spy_return), 6),
        "beats_spy": beats_spy,
    }


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman IC between predicted and actual returns. Canonical — import from here."""
    if len(y_true) < 5:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if np.isfinite(corr) else 0.0


def _empty_metrics(spy_return: float) -> dict:
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "n_trades": 0,
        "spy_return": round(float(spy_return), 6),
        "alpha": round(-float(spy_return), 6),
        "beats_spy": bool(False),
    }
