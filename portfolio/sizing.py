"""
Module: portfolio/sizing.py
Purpose: Confidence-weighted position sizing (§9.2).
Phase: 4 — Signal Engine + Portfolio Logic
Dependencies: config/thresholds.py, utils/logger.py
Last modified: 2026-06-10
"""

from config.thresholds import MAX_POSITION_PCT, POSITION_SIZE_BASE
from utils.logger import get_logger

log = get_logger(__name__)


def position_size(
    base_size: float,
    confidence: float,
    portfolio_value: float,
) -> float:
    """
    Compute the dollar size for a new position (§9.2).

    multiplier = 0.5 + confidence  →  range [1.0, 1.5] for confidence in [0.5, 1.0]
    raw = base_size * multiplier
    capped = min(raw, 0.25 * portfolio_value)  — hard cap at 25%

    Args:
        base_size: Base dollar allocation before confidence adjustment
        confidence: Composite confidence score [0, 1]
        portfolio_value: Current total portfolio value

    Returns:
        Dollar size for the position, capped at MAX_POSITION_PCT of portfolio.
    """
    multiplier = POSITION_SIZE_BASE + confidence  # [0.5+0, 0.5+1] = [0.5, 1.5]
    raw = base_size * multiplier
    max_allowed = MAX_POSITION_PCT * portfolio_value
    final_size = min(raw, max_allowed)

    log.info(
        "position_size_computed",
        base_size=round(base_size, 2),
        confidence=round(confidence, 3),
        multiplier=round(multiplier, 3),
        raw_size=round(raw, 2),
        capped_at=round(max_allowed, 2),
        final_size=round(final_size, 2),
    )
    return final_size
