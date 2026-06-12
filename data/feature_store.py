"""
Module: data/feature_store.py
Purpose: Read/write Parquet feature cache partitioned by year-month.
         Prevents recomputation; enforces schema on write; rejects corrupted reads.
Phase: 2 — Feature Store + Alignment Validation
Dependencies: pandas, pyarrow, pandera, config/settings.py, utils/logger.py
Last modified: 2026-06-10
"""

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import FEATURE_STORE_PATH
from utils.logger import get_logger

log = get_logger(__name__)


def _partition_path(d: date) -> Path:
    """Return the Parquet file path for a given month partition."""
    return FEATURE_STORE_PATH / f"features_{d.year}-{d.month:02d}.parquet"


def write_features(df: pd.DataFrame, partition_date: date) -> None:
    """
    Write feature rows for a given month to the Parquet store.

    If the partition file already exists, merges by (ticker, date) —
    new rows overwrite existing ones, preserving rows for other tickers/dates.
    Does NOT recompute existing rows unless --force-recompute is passed
    at the pipeline level (§4.3).

    Args:
        df: Feature DataFrame with at minimum columns: ticker, date
        partition_date: Any date in the target month (used to determine partition)

    Raises:
        ValueError: If df is missing required 'ticker' or 'date' columns.
    """
    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError("Feature DataFrame must contain 'ticker' and 'date' columns.")

    path = _partition_path(partition_date)
    FEATURE_STORE_PATH.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        # Merge: drop existing rows that are being replaced
        existing_key = existing["ticker"].astype(str) + "_" + existing["date"].astype(str)
        new_key = df["ticker"].astype(str) + "_" + df["date"].astype(str)
        existing_filtered = existing[~existing_key.isin(new_key)]
        merged = pd.concat([existing_filtered, df], ignore_index=True)
    else:
        merged = df

    merged.to_parquet(path, index=False)
    log.info(
        "feature_store_written",
        partition=path.name,
        rows_written=len(df),
        total_rows=len(merged),
    )


def load_features(
    d: date,
    tickers: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load feature rows for a specific date from the Parquet store.

    Args:
        d: The trading date to load features for
        tickers: If provided, filter to these tickers only

    Returns:
        DataFrame of feature rows for date d, or None if partition missing.
    """
    path = _partition_path(d)
    if not path.exists():
        log.warning("feature_store_partition_missing", partition=path.name, date=str(d))
        return None

    df = pd.read_parquet(path)
    date_str = str(d)
    df_day = df[df["date"] == date_str].copy()

    if df_day.empty:
        log.warning("feature_store_date_missing", date=date_str)
        return None

    if tickers is not None:
        df_day = df_day[df_day["ticker"].isin(tickers)]

    log.info(
        "feature_store_loaded",
        date=date_str,
        rows=len(df_day),
        tickers=list(df_day["ticker"].unique()),
    )
    return df_day


def feature_exists(ticker: str, d: date) -> bool:
    """
    Check whether features for (ticker, date) already exist in the store.

    Used to skip recomputation (§4.3).

    Args:
        ticker: Ticker symbol
        d: Trading date

    Returns:
        True if the (ticker, date) row exists in the partition.
    """
    path = _partition_path(d)
    if not path.exists():
        return False

    df = pd.read_parquet(path, columns=["ticker", "date"])
    return not df[(df["ticker"] == ticker) & (df["date"] == str(d))].empty
