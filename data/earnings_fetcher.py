"""
Earnings calendar fetcher.
Primary: Finnhub /calendar/earnings
Fallback: yfinance Ticker.calendar
"""

import os
import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

FINNHUB_KEY = os.getenv('FINNHUB_API_KEY', '')


def get_next_earnings_date(
    ticker: str,
    from_date: date = None,
) -> Optional[date]:
    """
    Get next earnings date for ticker.
    Returns None if unavailable.
    """
    # ETFs and indices have no earnings dates
    # yfinance returns 404 for these — skip early
    _NO_EARNINGS = {
        'SPY', 'QQQ', 'IWM', 'DIA',
        'GLD', 'TLT', 'VXX', 'UVXY',
        'SQQQ', 'SPXS', 'SH', 'PSQ',
    }
    if ticker in _NO_EARNINGS:
        return None

    if from_date is None:
        from_date = date.today()

    to_date = from_date + timedelta(days=90)

    # Try Finnhub first
    if FINNHUB_KEY:
        try:
            url = 'https://finnhub.io/api/v1/calendar/earnings'
            params = {
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'symbol': ticker,
                'token': FINNHUB_KEY,
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                earnings = data.get('earningsCalendar', [])
                if earnings:
                    dates = sorted([
                        datetime.strptime(e['date'], '%Y-%m-%d').date()
                        for e in earnings
                        if e.get('date')
                    ])
                    future = [d for d in dates if d >= from_date]
                    if future:
                        return future[0]
        except Exception as e:
            logger.warning(f'Finnhub earnings failed {ticker}: {e}')

    # Fallback: yfinance
    try:
        import yfinance as yf
        ticker_obj = yf.Ticker(ticker)
        cal = ticker_obj.calendar
        if cal is not None and not cal.empty:
            if 'Earnings Date' in cal.index:
                ed = cal.loc['Earnings Date']
                if hasattr(ed, '__iter__'):
                    ed = ed.iloc[0]
                if pd.notna(ed):
                    if hasattr(ed, 'date'):
                        return ed.date()
                    return pd.Timestamp(ed).date()
    except Exception as e:
        logger.warning(f'yfinance earnings failed {ticker}: {e}')

    return None


def is_safe_to_trade(
    ticker: str,
    entry_date: date,
    hold_days: int = 5,
    buffer_days: int = 3,
) -> tuple[bool, Optional[date]]:
    """
    Check if safe to enter trade.
    Returns (is_safe, next_earnings_date).

    Blocks trade if earnings falls within:
      entry_date to entry_date + hold_days + buffer_days
    """
    danger_window_end = entry_date + timedelta(days=hold_days + buffer_days)

    next_earnings = get_next_earnings_date(ticker, from_date=entry_date)

    if next_earnings is None:
        # Unknown → allow but log warning
        logger.warning(f'{ticker}: earnings date unknown — proceeding')
        return True, None

    if next_earnings <= danger_window_end:
        logger.info(
            f'{ticker}: BLOCKED — earnings {next_earnings} within danger '
            f'window (hold+buffer={hold_days + buffer_days}d)'
        )
        return False, next_earnings

    return True, next_earnings


def build_earnings_cache(
    tickers: list,
    reference_date: date = None,
) -> dict:
    """
    Build earnings calendar cache for all tickers.
    Returns dict: ticker → next_earnings_date or None
    """
    if reference_date is None:
        reference_date = date.today()

    cache = {}
    for ticker in tickers:
        try:
            cache[ticker] = get_next_earnings_date(ticker, from_date=reference_date)
            time.sleep(0.5)  # rate limit
        except Exception as e:
            logger.warning(f'Cache build failed {ticker}: {e}')
            cache[ticker] = None

    logger.info(
        f'Earnings cache built: '
        f'{sum(1 for v in cache.values() if v)} / {len(cache)} dates found'
    )
    return cache
