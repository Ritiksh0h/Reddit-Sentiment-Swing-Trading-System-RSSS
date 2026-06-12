"""
Module: data/market_loader.py
Purpose: Download and cache adjusted daily OHLCV via yfinance (dev) or Polygon.io (prod).
         Handles split/dividend adjustment, missing-rate checks, and disk caching.
Phase: 1 — Data Pipeline
Dependencies: yfinance, config/settings.py, config/thresholds.py, utils/logger.py
Last modified: 2026-06-10
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import RAW_DATA_PATH
from config.thresholds import MARKET_DATA_MAX_MISSING_RATE
from utils.logger import get_logger

log = get_logger(__name__)

MARKET_CACHE_DIR: Path = RAW_DATA_PATH / "market"


def load_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Download adjusted OHLCV for a single ticker from yfinance.

    Caches results as parquet. Returns None if data is unavailable or
    if the missing-rate check fails (§15 hard stop).

    Note: auto_adjust=True is mandatory — failure to adjust corrupts all
    historical return calculations (§4.2).

    Args:
        ticker: Uppercase ticker symbol, e.g. "NVDA"
        start_date: ISO date "YYYY-MM-DD" (inclusive)
        end_date: ISO date "YYYY-MM-DD" (inclusive)
        force_refresh: If True, bypass cache and re-download

    Returns:
        DataFrame indexed by date with lowercase columns:
        [open, high, low, close, volume]. Returns None on failure.
    """
    MARKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = MARKET_CACHE_DIR / f"{ticker}_{start_date}_{end_date}.parquet"

    if cache_file.exists() and not force_refresh:
        log.info("market_data_cache_hit", ticker=ticker, file=str(cache_file))
        return pd.read_parquet(cache_file)

    log.info("market_data_downloading", ticker=ticker, start=start_date, end=end_date)
    try:
        # auto_adjust=True: adjusts for splits and dividends — MANDATORY
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            show_errors=False,
        )
    except Exception as e:
        log.error("market_data_download_failed", ticker=ticker, error=str(e))
        return None

    if raw is None or raw.empty:
        log.warning("market_data_empty", ticker=ticker, start=start_date, end=end_date)
        return None

    # Flatten MultiIndex columns if present (yfinance ≥0.2 may return them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    df = raw.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"

    # Required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            log.error(
                "market_data_missing_column",
                ticker=ticker,
                missing_col=col,
                available_cols=list(df.columns),
            )
            return None

    # Hard stop: reject if missing rate too high (§15)
    missing_pct = float(df[required_cols].isna().mean().mean())
    if missing_pct > MARKET_DATA_MAX_MISSING_RATE:
        log.error(
            "market_data_missing_rate_exceeded",
            ticker=ticker,
            missing_pct=round(missing_pct, 4),
            threshold=MARKET_DATA_MAX_MISSING_RATE,
        )
        return None

    df = df[required_cols]
    df.to_parquet(cache_file)
    log.info(
        "market_data_cached",
        ticker=ticker,
        rows=len(df),
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        path=str(cache_file),
    )
    return df


def load_ohlcv_batch(
    tickers: list[str],
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV for multiple tickers. Failed tickers are omitted (logged, not raised).

    Args:
        tickers: List of uppercase ticker symbols
        start_date: ISO date string (inclusive)
        end_date: ISO date string (inclusive)
        force_refresh: Bypass cache if True

    Returns:
        Dict mapping ticker → OHLCV DataFrame. Excludes failed tickers.
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = load_ohlcv(ticker, start_date, end_date, force_refresh=force_refresh)
        if df is not None:
            result[ticker] = df
        else:
            log.warning("market_data_excluded", ticker=ticker, reason="load_failed")

    log.info(
        "market_data_batch_complete",
        requested=len(tickers),
        loaded=len(result),
        failed=len(tickers) - len(result),
    )
    return result
