"""
Module: utils/time_utils.py
Purpose: Market calendar utilities, timezone conversions, trading-day arithmetic.
         Used everywhere return calculations cross day boundaries.
Phase: All
Dependencies: pandas-market-calendars, pytz
Last modified: 2026-06-10

IMPORTANT: Never compute returns across non-trading days (§4.2).
           All date arithmetic goes through this module.
"""

from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal
import pytz

from config.settings import MARKET_TIMEZONE

# Cache the NYSE calendar instance — construction is expensive
_NYSE: Optional[mcal.calendars.nyse.NYSEExchangeCalendar] = None


def _get_nyse() -> mcal.calendars.nyse.NYSEExchangeCalendar:
    """Return a cached NYSE calendar instance."""
    global _NYSE
    if _NYSE is None:
        _NYSE = mcal.get_calendar("NYSE")
    return _NYSE


def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Return NYSE trading days between start_date and end_date (inclusive).

    Args:
        start_date: ISO date string "YYYY-MM-DD"
        end_date: ISO date string "YYYY-MM-DD"

    Returns:
        DatetimeIndex of UTC-normalized trading day timestamps.

    Example:
        >>> days = get_trading_days("2024-01-01", "2024-01-10")
        >>> len(days)  # only trading days, not weekends/holidays
        7
    """
    nyse = _get_nyse()
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return mcal.date_range(schedule, frequency="1D")


def is_trading_day(d: date) -> bool:
    """
    Return True if d is a NYSE trading day.

    Args:
        d: Date to check

    Returns:
        True if NYSE was/is open on that date.
    """
    nyse = _get_nyse()
    schedule = nyse.schedule(
        start_date=d.isoformat(), end_date=d.isoformat()
    )
    return not schedule.empty


def next_trading_day(d: date, n: int = 1) -> date:
    """
    Return the nth NYSE trading day after d (not including d itself).

    Args:
        d: Starting date
        n: Number of trading days forward

    Returns:
        The nth trading day after d.

    Raises:
        ValueError: If n < 1 or no trading day found in lookahead window.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    # Look ahead far enough to skip weekends and holidays
    lookahead = d + pd.Timedelta(days=n * 2 + 10)
    days = get_trading_days(d.isoformat(), lookahead.isoformat())

    # Skip the first entry if it equals d
    candidate_dates = [ts.date() for ts in days if ts.date() > d]
    if len(candidate_dates) < n:
        raise ValueError(f"Could not find {n} trading days after {d}")
    return candidate_dates[n - 1]


def prev_trading_day(d: date, n: int = 1) -> date:
    """
    Return the nth NYSE trading day before d (not including d itself).

    Args:
        d: Starting date
        n: Number of trading days backward

    Returns:
        The nth trading day before d.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    lookback = d - pd.Timedelta(days=n * 2 + 10)
    days = get_trading_days(lookback.isoformat(), d.isoformat())
    candidate_dates = [ts.date() for ts in days if ts.date() < d]
    if len(candidate_dates) < n:
        raise ValueError(f"Could not find {n} trading days before {d}")
    return candidate_dates[-(n)]


def market_open_utc(d: date) -> datetime:
    """
    Return NYSE market open time on d as a UTC-aware datetime (09:30 ET).

    Args:
        d: Trading date

    Returns:
        UTC datetime of market open.
    """
    et_tz = pytz.timezone(MARKET_TIMEZONE)
    local = et_tz.localize(datetime(d.year, d.month, d.day, 9, 30, 0))
    return local.astimezone(timezone.utc)


def market_close_utc(d: date) -> datetime:
    """
    Return NYSE market close time on d as a UTC-aware datetime (16:00 ET).

    Args:
        d: Trading date

    Returns:
        UTC datetime of market close.
    """
    et_tz = pytz.timezone(MARKET_TIMEZONE)
    local = et_tz.localize(datetime(d.year, d.month, d.day, 16, 0, 0))
    return local.astimezone(timezone.utc)


def to_utc(ts: datetime) -> datetime:
    """
    Ensure a datetime is UTC-aware. Naive datetimes are assumed UTC.

    Args:
        ts: Input datetime (aware or naive)

    Returns:
        UTC-aware datetime.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def to_et_datetime(ts: datetime) -> datetime:
    """
    Convert any datetime to Eastern Time.

    Args:
        ts: Input datetime (aware or naive; naive assumed UTC)

    Returns:
        ET-aware datetime.
    """
    utc_ts = to_utc(ts)
    et_tz = pytz.timezone(MARKET_TIMEZONE)
    return utc_ts.astimezone(et_tz)


def unix_to_utc(unix_ts: int | float) -> datetime:
    """
    Convert a Unix timestamp (seconds since epoch) to UTC datetime.

    Args:
        unix_ts: Seconds since Unix epoch

    Returns:
        UTC-aware datetime.
    """
    return datetime.fromtimestamp(float(unix_ts), tz=timezone.utc)
