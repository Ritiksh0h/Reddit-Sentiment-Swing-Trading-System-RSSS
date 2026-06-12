"""
Module: features/market_features.py
Purpose: Compute per-ticker market technical features (§5.2) from OHLCV data.
         All computations use strictly prior-day data (enforced via alignment.py).
Phase: 2 — Feature Store + Alignment Validation
Dependencies: pandas-ta, numpy, features/alignment.py, utils/logger.py
Last modified: 2026-06-10
"""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from features.alignment import get_market_window
from utils.logger import get_logger

log = get_logger(__name__)

# Lookback needed for the longest indicator (SMA50 needs 50 days + buffer)
_MAX_LOOKBACK_DAYS = 60


def compute_market_features(
    market_df: pd.DataFrame,
    ticker: str,
    row_date: date,
) -> Optional[dict]:
    """
    Compute all §5.2 market technical features for (ticker, row_date).

    Uses the get_market_window call to enforce that no same-day or future
    data enters the feature computation.

    Args:
        market_df: Full OHLCV DataFrame indexed by date for this ticker
        ticker: Ticker symbol (for logging)
        row_date: Feature row date T

    Returns:
        Dict of feature values, or None if market data is missing for row_date.
    """
    if not _TA_AVAILABLE:
        log.error("pandas_ta_unavailable", ticker=ticker)
        return None

    # Get prior-day window — alignment enforced here
    window = get_market_window(market_df, row_date, lookback_days=_MAX_LOOKBACK_DAYS)

    if len(window) < 2:
        log.warning(
            "market_features_insufficient_data",
            ticker=ticker,
            date=str(row_date),
            available_rows=len(window),
        )
        return None

    # Also need the row_date close for breakout_flag and as the base for relative calcs
    dates = pd.to_datetime(market_df.index).normalize()
    row_ts = pd.Timestamp(row_date)
    today_mask = dates == row_ts
    if not today_mask.any():
        log.warning("market_features_missing_row_date", ticker=ticker, date=str(row_date))
        return None

    today = market_df[today_mask.values].iloc[0]
    close_T = float(today["close"])
    volume_T = float(today["volume"])

    # Build a combined series including T for indicator calculation (uses only prior for windows)
    # We compute indicators on window (strictly prior), then reference T separately
    close_series = window["close"].astype(float)
    high_series = window["high"].astype(float)
    low_series = window["low"].astype(float)
    volume_series = window["volume"].astype(float)

    # --- Relative Volume ---
    vol_20_mean = volume_series.tail(20).mean() if len(volume_series) >= 20 else volume_series.mean()
    relative_volume = volume_T / vol_20_mean if vol_20_mean > 0 else None

    # --- RSI (14) ---
    rsi_14: Optional[float] = None
    if len(close_series) >= 14:
        rsi_df = pd.concat([close_series, pd.Series([close_T])], ignore_index=True)
        rsi_series = ta.rsi(rsi_df, length=14)
        if rsi_series is not None and not rsi_series.empty:
            rsi_14 = float(rsi_series.iloc[-1])
            if np.isnan(rsi_14):
                rsi_14 = None

    # --- MACD (12, 26, 9) ---
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    if len(close_series) >= 26:
        full_close = pd.concat([close_series, pd.Series([close_T])], ignore_index=True)
        macd_df = ta.macd(full_close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            last = macd_df.iloc[-1]
            macd_line = _safe_float(last.get("MACD_12_26_9"))
            macd_signal = _safe_float(last.get("MACDs_12_26_9"))
            macd_hist = _safe_float(last.get("MACDh_12_26_9"))

    # --- ATR (14) ---
    atr_14: Optional[float] = None
    if len(window) >= 14:
        atr_series = ta.atr(
            pd.concat([high_series, pd.Series([float(today["high"])])], ignore_index=True),
            pd.concat([low_series, pd.Series([float(today["low"])])], ignore_index=True),
            pd.concat([close_series, pd.Series([close_T])], ignore_index=True),
            length=14,
        )
        if atr_series is not None and not atr_series.empty:
            atr_14 = _safe_float(atr_series.iloc[-1])

    # --- Returns ---
    returns_1d: Optional[float] = None
    returns_3d: Optional[float] = None
    returns_5d: Optional[float] = None
    if len(close_series) >= 1:
        prev_1 = float(close_series.iloc[-1])
        returns_1d = (close_T - prev_1) / prev_1 if prev_1 > 0 else None
    if len(close_series) >= 3:
        prev_3 = float(close_series.iloc[-3])
        returns_3d = (close_T - prev_3) / prev_3 if prev_3 > 0 else None
    if len(close_series) >= 5:
        prev_5 = float(close_series.iloc[-5])
        returns_5d = (close_T - prev_5) / prev_5 if prev_5 > 0 else None

    # --- SMA distances ---
    distance_from_20ma: Optional[float] = None
    distance_from_50ma: Optional[float] = None
    if len(close_series) >= 20:
        sma_20 = float(close_series.tail(20).mean())
        distance_from_20ma = (close_T - sma_20) / sma_20 if sma_20 > 0 else None
    if len(close_series) >= 50:
        sma_50 = float(close_series.tail(50).mean())
        distance_from_50ma = (close_T - sma_50) / sma_50 if sma_50 > 0 else None

    # --- Trend slope (10d linear regression) ---
    trend_slope_10d: Optional[float] = None
    if len(close_series) >= 10:
        y = close_series.tail(10).values.astype(float)
        x = np.arange(len(y), dtype=float)
        if not np.any(np.isnan(y)):
            slope = float(np.polyfit(x, y, 1)[0])
            trend_slope_10d = slope

    # --- Breakout flag ---
    breakout_flag: int = 0
    if len(close_series) >= 20:
        rolling_max_20 = float(close_series.tail(20).max())
        breakout_flag = 1 if close_T > rolling_max_20 else 0

    # --- Volume trend ---
    volume_trend: Optional[float] = None
    if len(volume_series) >= 5:
        vol_5_mean = float(volume_series.tail(5).mean())
        volume_trend = volume_T - vol_5_mean

    features = {
        "ticker": ticker,
        "date": str(row_date),
        "close": close_T,
        "volume": volume_T,
        "relative_volume": relative_volume,
        "rsi_14": rsi_14,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "atr_14": atr_14,
        "returns_1d": returns_1d,
        "returns_3d": returns_3d,
        "returns_5d": returns_5d,
        "distance_from_20ma": distance_from_20ma,
        "distance_from_50ma": distance_from_50ma,
        "trend_slope_10d": trend_slope_10d,
        "breakout_flag": breakout_flag,
        "volume_trend": volume_trend,
    }

    log.info(
        "market_features_computed",
        ticker=ticker,
        date=str(row_date),
        rsi_14=round(rsi_14, 2) if rsi_14 is not None else None,
        relative_volume=round(relative_volume, 3) if relative_volume is not None else None,
    )
    return features


def _safe_float(val) -> Optional[float]:
    """Return float or None if NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None
