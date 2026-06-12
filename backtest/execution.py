"""
Module: backtest/execution.py
Purpose: Slippage model, fee calculation, and liquidity constraints (§10.2).
         Random seed is fixed — same input must produce same output every run.
Phase: 5 — Backtesting Engine
Dependencies: config/thresholds.py, utils/logger.py
Last modified: 2026-06-10
"""

import random
from typing import Optional

from config.thresholds import (
    BACKTEST_FEE_PCT,
    BACKTEST_LIQUIDITY_PCT,
    BACKTEST_RANDOM_SEED,
    BACKTEST_SLIPPAGE_MAX,
    BACKTEST_SLIPPAGE_MIN,
)
from utils.logger import get_logger

log = get_logger(__name__)

# Fixed random state for deterministic backtest reproduction (§10.2)
_rng = random.Random(BACKTEST_RANDOM_SEED)


def reset_rng() -> None:
    """Reset the random number generator to the fixed seed. Call at backtest start."""
    _rng.seed(BACKTEST_RANDOM_SEED)


def apply_slippage(limit_price: float) -> float:
    """
    Apply random slippage to a limit price (§10.2).

    fill_price = limit_price * (1 + uniform(0.0005, 0.002))

    Args:
        limit_price: The intended entry/exit price

    Returns:
        Filled price after slippage.
    """
    slippage = _rng.uniform(BACKTEST_SLIPPAGE_MIN, BACKTEST_SLIPPAGE_MAX)
    fill_price = limit_price * (1 + slippage)
    return fill_price


def compute_fee(trade_value: float) -> float:
    """
    Compute round-trip fee for a trade (§10.2).

    fee = trade_value * 0.0005 (0.05% per leg)

    Args:
        trade_value: Notional trade value

    Returns:
        Fee in dollars.
    """
    return trade_value * BACKTEST_FEE_PCT


def apply_liquidity_constraint(
    trade_value: float,
    daily_volume: float,
    ticker: str,
) -> float:
    """
    Cap trade value at 1% of daily volume to model realistic fill sizes (§10.2).

    Args:
        trade_value: Intended trade value
        daily_volume: Ticker's daily dollar volume
        ticker: For logging

    Returns:
        Adjusted trade value (may be smaller than intended).
    """
    max_fill = daily_volume * BACKTEST_LIQUIDITY_PCT
    if trade_value > max_fill:
        log.info(
            "liquidity_constraint_applied",
            ticker=ticker,
            intended=round(trade_value, 2),
            capped_at=round(max_fill, 2),
        )
        return max_fill
    return trade_value


def simulate_fill(
    ticker: str,
    limit_price: float,
    intended_value: float,
    daily_volume: float,
) -> dict:
    """
    Simulate a realistic trade fill with slippage, fees, and liquidity constraints.

    Args:
        ticker: Ticker symbol
        limit_price: The intended fill price
        intended_value: Notional trade value before constraints
        daily_volume: Ticker's daily dollar volume for liquidity check

    Returns:
        Dict with: fill_price, fill_value, fee, net_cost, shares
    """
    # Apply liquidity constraint first
    fill_value = apply_liquidity_constraint(intended_value, daily_volume, ticker)
    # Apply slippage
    fill_price = apply_slippage(limit_price)
    # Compute shares
    shares = fill_value / fill_price
    # Compute fee
    fee = compute_fee(fill_value)

    return {
        "ticker": ticker,
        "fill_price": round(fill_price, 4),
        "fill_value": round(fill_value, 2),
        "fee": round(fee, 4),
        "net_cost": round(fill_value + fee, 2),
        "shares": round(shares, 6),
    }
