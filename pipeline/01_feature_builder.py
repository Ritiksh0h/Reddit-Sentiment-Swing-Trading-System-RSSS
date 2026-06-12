#!/usr/bin/env python3
"""
Module: pipeline/01_feature_builder.py
Purpose: Build the (ticker, date) feature matrix from Reddit sentiment data
         + yfinance market data. Primary input for all downstream scripts.

         Reddit features use strict < 09:30 ET cutoff on date T.
         Market features use same-day close (end-of-day decision).
         Target labels use strictly future prices (T+N trading days).

Phase: 1 — Research Pipeline
Input:  data/raw/merged_with_sentiment.parquet  (880,070 rows)
Output: data/features/features.parquet

Usage:
    python pipeline/01_feature_builder.py
    python pipeline/01_feature_builder.py --debug           # first 1000 rows
    python pipeline/01_feature_builder.py --force-recompute # bypass cache
Last modified: 2026-06-11
"""

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import pandas_market_calendars as mcal
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    SENTIMENT_PARQUET,
    FEATURES_PARQUET,
    DATA_PROC,
    TRAIN_END,
    TEST_START,
    MARKET_TZ,
    CUTOFF_HOUR,
    CUTOFF_MIN,
)
from config.thresholds import (
    MIN_POST_COUNT,
    MIN_CONFIDENCE,
    MAX_RETURN_WINSORISE,
)
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCLUDED_TICKERS: frozenset[str] = frozenset({"GME", "AAPL", "GOOG", "AMZN", "HOOD"})
FOCUS_TICKERS_FILE: Path = Path(__file__).parent.parent / "config" / "focus_tickers.txt"
MARKET_CACHE_DIR: Path = DATA_PROC / "market"

# Feature columns in output order
REDDIT_FEATURE_COLS: list[str] = [
    "post_count_1d", "post_count_3d", "post_count_7d",
    "unique_authors_1d", "total_upvotes_1d", "total_comments_1d",
    "mention_growth_1d", "mention_growth_7d",
    "avg_sentiment_1d", "avg_sentiment_3d", "weighted_sentiment",
    "sentiment_std", "sentiment_accel", "bullish_ratio", "avg_sentiment_hc",
]
MARKET_FEATURE_COLS: list[str] = [
    "close", "volume",
    "returns_1d", "returns_5d", "returns_20d",
    "rsi_14", "atr_14", "relative_volume",
    "dist_from_20ma", "dist_from_50ma",
]
TARGET_COLS: list[str] = ["target_return_5d", "target_return_3d", "target_return_1d", "target_return_10d"]
FINAL_COL_ORDER: list[str] = (
    ["ticker", "date"]
    + REDDIT_FEATURE_COLS
    + MARKET_FEATURE_COLS
    + TARGET_COLS
    + ["split"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_focus_tickers() -> list[str]:
    """Load focus tickers from config/focus_tickers.txt, excluding the excluded set."""
    with open(FOCUS_TICKERS_FILE) as f:
        tickers = [
            line.strip().upper()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
    return [t for t in tickers if t not in EXCLUDED_TICKERS]


def get_trading_days(start: str, end: str) -> list[date]:
    """Return NYSE trading days between start and end (inclusive) as date objects."""
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start, end_date=end)
    dt_index = mcal.date_range(sched, frequency="1D")
    return [ts.date() for ts in dt_index]


def make_cutoffs_ns(trading_days: list[date]) -> np.ndarray:
    """
    Pre-compute 09:30 ET cutoffs as int64 nanoseconds UTC for all trading days.
    Vectorized over dates for fast searchsorted operations.
    """
    et_tz = pytz.timezone(MARKET_TZ)
    cutoffs = []
    for d in trading_days:
        dt_et = et_tz.localize(datetime(d.year, d.month, d.day, CUTOFF_HOUR, CUTOFF_MIN, 0))
        dt_utc = dt_et.astimezone(timezone.utc)
        cutoffs.append(int(dt_utc.timestamp() * 1_000_000_000))
    return np.array(cutoffs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Reddit Feature Computation
# ---------------------------------------------------------------------------

def compute_reddit_features(
    reddit_df: pd.DataFrame,
    tickers: list[str],
    trading_days: list[date],
) -> pd.DataFrame:
    """
    Compute per-(ticker, date) Reddit sentiment and attention features.

    Windows use strict < cutoff (09:30 ET on date T). Never <=.
    Rows where post_count_1d < MIN_POST_COUNT are dropped here.
    Missing sentiment is left as NaN — never filled (§3.3).

    Args:
        reddit_df: Deduplicated Reddit posts with UTC timestamps
        tickers: Focus ticker list
        trading_days: NYSE trading days to compute features for

    Returns:
        DataFrame with one row per (ticker, date) that passes MIN_POST_COUNT.
    """
    cutoffs_arr = make_cutoffs_ns(trading_days)

    # Convert timestamp to int64 ns once for the full DataFrame
    reddit_df = reddit_df.copy()
    reddit_df["ts_ns"] = pd.to_datetime(reddit_df["timestamp"], utc=True).astype(np.int64)

    ns_per_hour = np.int64(3600 * 1_000_000_000)

    all_rows: list[dict] = []

    for ticker in tickers:
        ticker_posts = (
            reddit_df[reddit_df["ticker"] == ticker]
            .sort_values("ts_ns")
            .reset_index(drop=True)
        )
        if ticker_posts.empty:
            log.warning("reddit_no_posts", ticker=ticker)
            continue

        ts_ns = ticker_posts["ts_ns"].values  # sorted int64 array

        for day, cutoff_ns in zip(trading_days, cutoffs_arr):
            # Compute window boundaries in nanoseconds
            end = int(np.searchsorted(ts_ns, cutoff_ns, side="left"))  # strict <
            s1d = int(np.searchsorted(ts_ns, cutoff_ns - 24 * ns_per_hour, side="left"))
            s3d = int(np.searchsorted(ts_ns, cutoff_ns - 72 * ns_per_hour, side="left"))
            s7d = int(np.searchsorted(ts_ns, cutoff_ns - 168 * ns_per_hour, side="left"))

            count_1d = end - s1d
            if count_1d < MIN_POST_COUNT:
                continue

            count_3d = end - s3d
            count_7d = end - s7d

            p1d = ticker_posts.iloc[s1d:end]
            p3d = ticker_posts.iloc[s3d:end]

            # Attention features
            total_upvotes_1d = float(p1d["score"].sum())
            total_comments_1d = float(p1d["num_comments"].sum())
            unique_authors_1d = int(p1d["post_id"].nunique())  # proxy for unique authors
            mention_growth_1d = count_1d / (count_3d + 1)
            mention_growth_7d = count_1d / (count_7d + 1)

            # Sentiment features (drop NaN — never fill per §3.3)
            v1d = p1d.dropna(subset=["sentiment_score"])
            v3d = p3d.dropna(subset=["sentiment_score"])

            avg_s1d: Optional[float] = float(v1d["sentiment_score"].mean()) if not v1d.empty else None
            avg_s3d: Optional[float] = float(v3d["sentiment_score"].mean()) if not v3d.empty else None

            # Weighted sentiment: sum(score * log1p(clip(upvotes, 0))) / post_count_1d
            weighted_sentiment: Optional[float] = None
            if not v1d.empty:
                w = np.log1p(v1d["score"].clip(lower=0).values.astype(float))
                weighted_sentiment = float((v1d["sentiment_score"].values * w).sum()) / count_1d

            sentiment_std: Optional[float] = float(v1d["sentiment_score"].std()) if len(v1d) > 1 else None
            sentiment_accel: Optional[float] = (
                (avg_s1d - avg_s3d) if avg_s1d is not None and avg_s3d is not None else None
            )

            # Bullish ratio
            bullish_ratio: Optional[float] = None
            if count_1d > 0:
                n_pos = (p1d["sentiment_label"] == "positive").sum()
                bullish_ratio = float(n_pos) / count_1d

            # High-confidence sentiment (sentiment_conf >= MIN_CONFIDENCE)
            hc = p1d[p1d["sentiment_conf"] >= MIN_CONFIDENCE]
            avg_sentiment_hc: Optional[float] = (
                float(hc["sentiment_score"].mean()) if not hc.empty else None
            )

            all_rows.append({
                "ticker": ticker,
                "date": day,
                "post_count_1d": count_1d,
                "post_count_3d": count_3d,
                "post_count_7d": count_7d,
                "unique_authors_1d": unique_authors_1d,
                "total_upvotes_1d": total_upvotes_1d,
                "total_comments_1d": total_comments_1d,
                "mention_growth_1d": mention_growth_1d,
                "mention_growth_7d": mention_growth_7d,
                "avg_sentiment_1d": avg_s1d,
                "avg_sentiment_3d": avg_s3d,
                "weighted_sentiment": weighted_sentiment,
                "sentiment_std": sentiment_std,
                "sentiment_accel": sentiment_accel,
                "bullish_ratio": bullish_ratio,
                "avg_sentiment_hc": avg_sentiment_hc,
            })

        n_rows = sum(1 for r in all_rows if r["ticker"] == ticker)
        log.info("reddit_features_computed", ticker=ticker, rows=n_rows)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Market Data + Feature Computation
# ---------------------------------------------------------------------------

def load_market_data(
    ticker: str,
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Download adjusted OHLCV via yfinance, caching to data/processed/market/.
    auto_adjust=True is mandatory — failure to adjust corrupts return calculations.

    Args:
        ticker: Uppercase ticker symbol
        start_date: "YYYY-MM-DD" (inclusive; fetch extra buffer for rolling indicators)
        end_date: "YYYY-MM-DD" (inclusive)
        force_refresh: Bypass cache if True

    Returns:
        DataFrame indexed by date with columns [open, high, low, close, volume].
        None on failure.
    """
    MARKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = MARKET_CACHE_DIR / f"{ticker}_{start_date}_{end_date}.parquet"

    if cache.exists() and not force_refresh:
        log.info("market_cache_hit", ticker=ticker)
        return pd.read_parquet(cache)

    log.info("market_downloading", ticker=ticker, start=start_date, end=end_date)
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        log.error("market_download_failed", ticker=ticker, error=str(e))
        return None

    if raw is None or raw.empty:
        log.warning("market_empty", ticker=ticker)
        return None

    # Flatten MultiIndex if present (yfinance ≥0.2)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    df.to_parquet(cache)
    return df


def compute_market_features(
    ticker: str,
    ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all §5.2 market technical features + forward return labels.

    Uses vectorized pandas-ta operations on the full OHLCV DataFrame.
    Forward returns (targets) use shift(-N) — strictly future prices.

    Args:
        ticker: Ticker symbol (for logging)
        ohlcv: Full OHLCV DataFrame indexed by date

    Returns:
        DataFrame with market features + target labels, indexed by date.
    """
    df = ohlcv.copy()

    # --- Price returns (lookback) ---
    df["returns_1d"] = df["close"].pct_change(1)
    df["returns_5d"] = df["close"].pct_change(5)
    df["returns_20d"] = df["close"].pct_change(20)

    # --- Technical indicators (pandas-ta) ---
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    raw_atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_14"] = raw_atr / df["close"]  # normalized by close

    # --- Volume ---
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["relative_volume"] = df["volume"] / df["vol_ma_20"]

    # --- Moving average distances ---
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["dist_from_20ma"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["dist_from_50ma"] = (df["close"] - df["sma_50"]) / df["sma_50"]

    # --- Forward return labels (strictly future — shift by -N) ---
    df["target_return_1d"] = df["close"].shift(-1) / df["close"] - 1
    df["target_return_3d"] = df["close"].shift(-3) / df["close"] - 1
    df["target_return_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["target_return_10d"] = df["close"].shift(-10) / df["close"] - 1

    df["ticker"] = ticker
    df["date"] = df.index.date

    keep_cols = (
        ["ticker", "date", "close", "volume"]
        + ["returns_1d", "returns_5d", "returns_20d"]
        + ["rsi_14", "atr_14", "relative_volume"]
        + ["dist_from_20ma", "dist_from_50ma"]
        + ["target_return_5d", "target_return_3d", "target_return_1d", "target_return_10d"]
    )
    result = df[keep_cols].copy()
    log.info("market_features_computed", ticker=ticker, rows=len(result))
    return result


def load_all_market_features(
    tickers: list[str],
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load and compute market features for all tickers, returning a combined DataFrame.
    Fetches extra buffer (60 extra calendar days) to ensure rolling indicators have data.
    """
    # Extra buffer for 50-day SMA + target label lookback
    start_dt = pd.Timestamp(start_date) - pd.Timedelta(days=90)
    start_buffered = start_dt.strftime("%Y-%m-%d")
    # Extra buffer at end for 5-day forward returns
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=15)
    end_buffered = end_dt.strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        ohlcv = load_market_data(ticker, start_buffered, end_buffered, force_refresh)
        if ohlcv is None:
            log.warning("market_data_skipped", ticker=ticker, reason="load_failed")
            continue
        feat = compute_market_features(ticker, ohlcv)
        frames.append(feat)

    if not frames:
        raise RuntimeError("No market data loaded for any ticker.")

    combined = pd.concat(frames, ignore_index=True)
    # Normalise date column to string "YYYY-MM-DD" for consistent join key
    combined["date"] = combined["date"].astype(str)
    log.info("market_features_all_done", tickers=len(frames), rows=len(combined))
    return combined


# ---------------------------------------------------------------------------
# Quality Filter + Leakage Validation
# ---------------------------------------------------------------------------

class DataIntegrityError(Exception):
    """Raised when a leakage check fails. Pipeline must halt."""


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply §5.4 quality filters. Log every excluded row's reason.

    Filters:
    - post_count_1d < MIN_POST_COUNT (already excluded during Reddit feature build, kept as guard)
    - Any market feature is NaN
    - target_return_5d is NaN (end of series)
    - |target_return_5d| > MAX_RETURN_WINSORISE (winsorise at 50% — removes halts/earnings gaps)
    - Ticker in EXCLUDED_TICKERS (guard)
    """
    original = len(df)

    # Guard: excluded tickers
    mask_excl = df["ticker"].isin(EXCLUDED_TICKERS)
    if mask_excl.any():
        log.info("quality_filter_excluded_tickers", dropped=int(mask_excl.sum()))
        df = df[~mask_excl]

    # Guard: post_count gate
    mask_pc = df["post_count_1d"] < MIN_POST_COUNT
    if mask_pc.any():
        log.info("quality_filter_post_count", dropped=int(mask_pc.sum()))
        df = df[~mask_pc]

    # Drop NaN target
    mask_notgt = df["target_return_5d"].isna()
    if mask_notgt.any():
        log.info("quality_filter_target_nan", dropped=int(mask_notgt.sum()))
        df = df[~mask_notgt]

    # Winsorise extreme returns (halts, acquisition announcements, data errors)
    mask_extreme = df["target_return_5d"].abs() > MAX_RETURN_WINSORISE
    if mask_extreme.any():
        log.info("quality_filter_extreme_returns", dropped=int(mask_extreme.sum()),
                 threshold=MAX_RETURN_WINSORISE)
        df = df[~mask_extreme]

    # Drop rows with NaN in any required market feature
    market_required = ["close", "volume", "returns_1d", "returns_5d", "returns_20d",
                       "rsi_14", "atr_14", "relative_volume", "dist_from_20ma", "dist_from_50ma"]
    mask_mkt = df[market_required].isna().any(axis=1)
    if mask_mkt.any():
        log.info("quality_filter_market_nan", dropped=int(mask_mkt.sum()))
        df = df[~mask_mkt]

    log.info(
        "quality_filters_applied",
        original=original,
        remaining=len(df),
        dropped=original - len(df),
    )
    return df.reset_index(drop=True)


def validate_no_leakage(df: pd.DataFrame) -> None:
    """
    Post-hoc integrity checks on the feature matrix.

    Since leakage is prevented during computation (strict < cutoff in
    compute_reddit_features), this validates the output invariants:
    1. Targets are not all NaN
    2. Train data ends before TEST_START
    3. Test data starts at or after TEST_START

    Raises DataIntegrityError if any check fails.
    """
    if df["target_return_5d"].isna().all():
        raise DataIntegrityError("All target_return_5d values are NaN — pipeline failed.")

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    if not train.empty:
        max_train_date = pd.Timestamp(train["date"].max())
        if max_train_date >= pd.Timestamp(TEST_START):
            raise DataIntegrityError(
                f"Train data contains dates >= TEST_START: max={max_train_date.date()}"
            )

    if not test.empty:
        min_test_date = pd.Timestamp(test["date"].min())
        if min_test_date < pd.Timestamp(TEST_START):
            raise DataIntegrityError(
                f"Test data contains dates < TEST_START: min={min_test_date.date()}"
            )

    log.info(
        "leakage_validation_passed",
        train_rows=len(train),
        test_rows=len(test),
        train_end=train["date"].max() if not train.empty else "N/A",
        test_start=test["date"].min() if not test.empty else "N/A",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_features(
    debug: bool = False,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Build and return the complete feature matrix.

    Caches to FEATURES_PARQUET. Loads from cache unless force_recompute=True.

    Args:
        debug: If True, use only first 1000 rows of Reddit data (fast sanity check)
        force_recompute: Bypass cache and recompute everything

    Returns:
        Feature DataFrame ready for downstream scripts.
    """
    if FEATURES_PARQUET.exists() and not force_recompute and not debug:
        log.info("feature_cache_hit", path=str(FEATURES_PARQUET))
        return pd.read_parquet(FEATURES_PARQUET)

    # --- 1. Load Reddit data ---
    log.info("loading_reddit_parquet", path=str(SENTIMENT_PARQUET))
    reddit_raw = pd.read_parquet(SENTIMENT_PARQUET)
    log.info("reddit_loaded", rows=len(reddit_raw), columns=list(reddit_raw.columns))

    if debug:
        reddit_raw = reddit_raw.head(1000)
        log.info("debug_mode", rows=len(reddit_raw))

    # --- 2. Deduplicate on (post_id, ticker) — keep highest-confidence match ---
    reddit_raw = (
        reddit_raw
        .sort_values("confidence", ascending=False)
        .drop_duplicates(subset=["post_id", "ticker"])
        .reset_index(drop=True)
    )
    log.info("reddit_deduplicated", rows=len(reddit_raw))

    # --- 3. Filter to focus tickers ---
    tickers = load_focus_tickers()
    reddit_raw = reddit_raw[reddit_raw["ticker"].isin(tickers)].reset_index(drop=True)
    log.info("reddit_focus_filtered", tickers=len(tickers), rows=len(reddit_raw))

    if reddit_raw.empty:
        raise RuntimeError("No rows remain after filtering to focus tickers.")

    # --- 4. Determine date range from data ---
    ts_min = pd.to_datetime(reddit_raw["timestamp"], utc=True).min()
    ts_max = pd.to_datetime(reddit_raw["timestamp"], utc=True).max()
    data_start = ts_min.date().isoformat()
    data_end = ts_max.date().isoformat()
    log.info("date_range", start=data_start, end=data_end)

    trading_days = get_trading_days(data_start, data_end)
    log.info("trading_days", count=len(trading_days))

    # --- 5. Compute Reddit features ---
    log.info("computing_reddit_features")
    reddit_features = compute_reddit_features(reddit_raw, tickers, trading_days)
    log.info("reddit_features_done", rows=len(reddit_features))

    if reddit_features.empty:
        raise RuntimeError("Reddit feature computation returned no rows.")

    # Normalise date to string for join
    reddit_features["date"] = reddit_features["date"].astype(str)

    # --- 6. Load and compute market features ---
    log.info("computing_market_features")
    market_features = load_all_market_features(
        tickers,
        start_date=data_start,
        end_date=data_end,
        force_refresh=force_recompute,
    )

    # --- 7. Join Reddit + market features on (ticker, date) ---
    combined = pd.merge(
        reddit_features,
        market_features,
        on=["ticker", "date"],
        how="inner",
    )
    log.info("features_joined", rows=len(combined))

    # --- 8. Quality filters ---
    combined = apply_quality_filters(combined)

    # --- 9. Add train/test split column ---
    combined["split"] = np.where(
        combined["date"] < TEST_START, "train", "test"
    )
    log.info(
        "split_added",
        train=int((combined["split"] == "train").sum()),
        test=int((combined["split"] == "test").sum()),
    )

    # --- 10. Reorder columns ---
    available_cols = [c for c in FINAL_COL_ORDER if c in combined.columns]
    extra_cols = [c for c in combined.columns if c not in FINAL_COL_ORDER]
    combined = combined[available_cols + extra_cols]

    # --- 11. Leakage validation ---
    validate_no_leakage(combined)

    # --- 12. Save (unless debug mode) ---
    if not debug:
        FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(FEATURES_PARQUET, index=False)
        log.info(
            "features_saved",
            path=str(FEATURES_PARQUET),
            rows=len(combined),
            tickers=combined["ticker"].nunique(),
            date_range=f"{combined['date'].min()} → {combined['date'].max()}",
            train_rows=int((combined["split"] == "train").sum()),
            test_rows=int((combined["split"] == "test").sum()),
        )
    else:
        log.info("debug_mode_not_saving", rows=len(combined))

    return combined


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(description="Phase 1 — Feature Builder")
    parser.add_argument("--debug", action="store_true",
                        help="Run on first 1000 rows only (fast sanity check)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore cache and recompute all features")
    args = parser.parse_args()

    log.info("feature_builder_start", debug=args.debug, force_recompute=args.force_recompute)

    df = build_features(debug=args.debug, force_recompute=args.force_recompute)

    # Summary output
    print("\n=== FEATURE BUILD SUMMARY ===")
    print(f"  Rows:     {len(df):,}")
    print(f"  Tickers:  {df['ticker'].nunique()}")
    print(f"  Dates:    {df['date'].min()} → {df['date'].max()}")
    print(f"  Train:    {(df['split']=='train').sum():,} rows")
    print(f"  Test:     {(df['split']=='test').sum():,} rows")
    print(f"  Columns:  {len(df.columns)}")
    print(f"  NaN pct (reddit feats): {df[REDDIT_FEATURE_COLS].isna().mean().mean():.1%}")
    print(f"  Target mean (train):    {df[df['split']=='train']['target_return_5d'].mean():.4f}")
    print(f"  Target std  (train):    {df[df['split']=='train']['target_return_5d'].std():.4f}")
    if not args.debug:
        print(f"  Saved to: {FEATURES_PARQUET}")
    print("=" * 30)


if __name__ == "__main__":
    main()
