"""
Module: signals/ranking.py
Purpose: Rank multiple BUY signals and select top-N for the portfolio engine (§8).
Phase: 4 — Signal Engine + Portfolio Logic
Dependencies: config/thresholds.py, utils/logger.py
Last modified: 2026-06-10
"""

from config.thresholds import MAX_BUY_SIGNALS, RANK_WEIGHT_ACCEL, RANK_WEIGHT_CONFIDENCE, RANK_WEIGHT_RETURN
from utils.logger import get_logger

log = get_logger(__name__)


def score_signal(pred_5d: float, confidence: float, sentiment_accel: float) -> float:
    """
    Compute the ranking score for a BUY signal.

    score = (0.5 * pred_5d) + (0.3 * confidence) + (0.2 * sentiment_accel)

    Args:
        pred_5d: Model's 5-day return prediction
        confidence: Composite confidence score [0, 1]
        sentiment_accel: Sentiment acceleration (1d avg - 3d avg)

    Returns:
        Scalar ranking score (higher = better).
    """
    return (
        RANK_WEIGHT_RETURN * pred_5d
        + RANK_WEIGHT_CONFIDENCE * confidence
        + RANK_WEIGHT_ACCEL * sentiment_accel
    )


def rank_and_select(signals: list[dict]) -> list[dict]:
    """
    Rank BUY signals by score and return the top MAX_BUY_SIGNALS.

    Only processes signals where signal == "BUY". HOLD/AVOID signals are
    passed through without ranking.

    Ranking is deterministic — same input → same output (§10.2).

    Args:
        signals: List of signal dicts from generator.generate_signal()

    Returns:
        Top MAX_BUY_SIGNALS BUY signals, sorted descending by score.
    """
    buy_signals = [s for s in signals if s.get("signal") == "BUY"]

    for s in buy_signals:
        s["rank_score"] = score_signal(
            pred_5d=s.get("prediction_5d", 0.0) or 0.0,
            confidence=s.get("confidence", 0.0) or 0.0,
            sentiment_accel=s.get("sentiment_acceleration", 0.0) or 0.0,
        )

    ranked = sorted(buy_signals, key=lambda x: x["rank_score"], reverse=True)
    top_n = ranked[:MAX_BUY_SIGNALS]

    log.info(
        "signals_ranked",
        total_buy_signals=len(buy_signals),
        selected=len(top_n),
        top_tickers=[s["ticker"] for s in top_n],
    )
    return top_n
