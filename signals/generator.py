"""
Module: signals/generator.py
Purpose: Generate BUY / HOLD / AVOID signals from model predictions (§8).
Phase: 4 — Signal Engine + Portfolio Logic
Dependencies: config/thresholds.py, utils/logger.py
Last modified: 2026-06-10
"""

from datetime import datetime, timezone
from typing import Optional

from config.thresholds import (
    MIN_CONFIDENCE_FOR_ANY_SIGNAL,
    SIGNAL_BUY_MIN_PRED_5D,
    SIGNAL_AVOID_MAX_PRED_5D,
    SIGNAL_CONFIDENCE_THRESHOLD,
    SIGNAL_MIN_RVOL,
    SIGNAL_MIN_SENTIMENT_ACCEL,
)
from utils.logger import get_logger

log = get_logger(__name__)


def generate_signal(
    ticker: str,
    pred_1d: Optional[float],
    pred_3d: Optional[float],
    pred_5d: Optional[float],
    confidence: float,
    sentiment_accel: Optional[float],
    rvol: Optional[float],
    model_version: str,
) -> dict:
    """
    Generate a BUY / HOLD / AVOID signal for a single ticker.

    Logic per §8:
      BUY  → pred_5d > 0.03 AND confidence > 0.70 AND sentiment_accel > 0 AND rvol > 1.2
      AVOID→ pred_5d < -0.03 AND confidence > 0.70
      HOLD → everything else, including confidence < 0.50

    Args:
        ticker: Uppercase ticker symbol
        pred_1d: Model's 1-day return prediction
        pred_3d: Model's 3-day return prediction
        pred_5d: Model's 5-day return prediction
        confidence: Composite confidence score [0, 1] (§7)
        sentiment_accel: avg_sentiment_1d - avg_sentiment_3d
        rvol: Relative volume (volume[T] / mean(volume[T-20:T]))
        model_version: Identifier of the model that produced predictions

    Returns:
        Signal dict with keys: ticker, signal, prediction_1d, prediction_3d,
        prediction_5d, confidence, model_version, generated_at.
    """
    # Hard gate: confidence too low for any non-HOLD signal
    if confidence < MIN_CONFIDENCE_FOR_ANY_SIGNAL:
        signal = "HOLD"
    elif (
        pred_5d is not None
        and pred_5d > SIGNAL_BUY_MIN_PRED_5D
        and confidence > SIGNAL_CONFIDENCE_THRESHOLD
        and (sentiment_accel is not None and sentiment_accel > SIGNAL_MIN_SENTIMENT_ACCEL)
        and (rvol is not None and rvol > SIGNAL_MIN_RVOL)
    ):
        signal = "BUY"
    elif (
        pred_5d is not None
        and pred_5d < SIGNAL_AVOID_MAX_PRED_5D
        and confidence > SIGNAL_CONFIDENCE_THRESHOLD
    ):
        signal = "AVOID"
    else:
        signal = "HOLD"

    result = {
        "ticker": ticker,
        "signal": signal,
        "prediction_1d": pred_1d,
        "prediction_3d": pred_3d,
        "prediction_5d": pred_5d,
        "confidence": confidence,
        "model_version": model_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        "signal_generated",
        ticker=ticker,
        signal=signal,
        pred_5d=round(pred_5d, 4) if pred_5d is not None else None,
        confidence=round(confidence, 3),
    )
    return result
