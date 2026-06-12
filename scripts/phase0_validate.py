"""
Module: scripts/phase0_validate.py
Purpose: Phase 0 signal validation — check whether Reddit sentiment has a
         measurable IC (≥ 0.03) against next-day returns before building infrastructure.

         If IC < 0.03 across all tickers: ABORT — thesis is wrong.
         If IC ≥ 0.03: PROCEED to Phase 1.

Phase: 0 — Signal Validation
Dependencies: datasets (HuggingFace), transformers (FinBERT), yfinance,
              config/settings.py, config/thresholds.py, data/market_loader.py,
              utils/logger.py, utils/time_utils.py
Last modified: 2026-06-10

Usage:
    python scripts/phase0_validate.py
    python scripts/phase0_validate.py --dataset RomanBlanco/reddit_wsb_2021
    python scripts/phase0_validate.py --dataset local --local-path data/raw/wsb_posts.csv
    python scripts/phase0_validate.py --tickers NVDA TSLA GME
    python scripts/phase0_validate.py --debug     # dry-run on 500 posts

Dataset candidates (ranked by fit):
    1. Lelon/reddit-wsb-posts       — Pushshift WSB 2012-2022 (best coverage)
    2. RomanBlanco/reddit_wsb_2021  — WSB GME squeeze period (narrow)
    3. SocialGrep/one-million-reddit-comments  — mixed subreddits (comments only)

    Run `python scripts/phase0_validate.py --list-datasets` to see all candidates.
"""

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root to path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    PHASE0_TICKERS,
    PHASE0_HF_DATASETS,
    PHASE0_RESULTS_PATH,
    FINBERT_MODEL_NAME,
    FINBERT_BATCH_SIZE,
    FINBERT_DEVICE,
    RAW_DATA_PATH,
    MONITORED_SUBREDDITS,
)
from config.thresholds import (
    IC_ABORT_THRESHOLD,
    MIN_POST_COUNT_1D,
    FINBERT_MAX_FAILURE_RATE,
    MARKET_DATA_MAX_MISSING_RATE,
)
from data.market_loader import load_ohlcv_batch
from utils.logger import get_logger
from utils.time_utils import get_trading_days, unix_to_utc, market_open_utc

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 0: Reddit sentiment IC validation"
    )
    parser.add_argument(
        "--dataset",
        default=PHASE0_HF_DATASETS[0]["id"],
        help=f"HuggingFace dataset ID or 'local'. Default: {PHASE0_HF_DATASETS[0]['id']}",
    )
    parser.add_argument(
        "--local-path",
        default=None,
        help="Path to local CSV/Parquet if --dataset=local",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=PHASE0_TICKERS,
        help="Tickers to validate. Default: Phase 0 ticker list from settings.py",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start of analysis window. Default: 2020-01-01",
    )
    parser.add_argument(
        "--end-date",
        default="2022-12-31",
        help="End of analysis window. Default: 2022-12-31",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dry-run on first 500 posts only (fast sanity check)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List known HuggingFace dataset candidates and exit",
    )
    parser.add_argument(
        "--skip-finbert",
        action="store_true",
        help="Skip FinBERT; use upvote-weighted post count only (fast baseline IC check)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_hf_dataset(dataset_id: str, debug: bool = False) -> pd.DataFrame:
    """
    Load a HuggingFace Reddit dataset and normalise to a common schema.

    Returns DataFrame with columns:
        post_id, subreddit, title, body, upvotes, comment_count,
        author, created_utc_unix (float), timestamp (UTC datetime str)

    Args:
        dataset_id: HuggingFace dataset identifier
        debug: If True, load only 500 rows

    Returns:
        Normalised DataFrame.

    Raises:
        RuntimeError: If the dataset cannot be loaded or required columns are missing.
    """
    log.info("hf_dataset_loading", dataset_id=dataset_id, debug=debug)

    try:
        from datasets import load_dataset as hf_load
    except ImportError as e:
        raise RuntimeError(
            "HuggingFace `datasets` library not installed. "
            "Run: pip install datasets"
        ) from e

    # Find the schema mapping for this dataset
    schema_map: Optional[dict] = None
    for candidate in PHASE0_HF_DATASETS:
        if candidate["id"] == dataset_id:
            schema_map = candidate
            break

    try:
        raw = hf_load(dataset_id, split="train", streaming=False)
        df = raw.to_pandas()
        log.info("hf_dataset_loaded", rows=len(df), columns=list(df.columns))
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_id}': {e}") from e

    if debug:
        df = df.head(500)
        log.info("debug_mode_active", rows_truncated_to=500)

    return _normalise_reddit_df(df, schema_map)


def load_local_dataset(path: str, debug: bool = False) -> pd.DataFrame:
    """
    Load a local CSV or Parquet Reddit dataset.

    Expects columns: post_id (or id), title, selftext (or body),
    score (or upvotes), created_utc, subreddit, author.

    Args:
        path: Path to CSV or Parquet file
        debug: If True, load only 500 rows

    Returns:
        Normalised DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")

    log.info("local_dataset_loading", path=path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif p.suffix in {".csv", ".gz"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .csv or .parquet")

    if debug:
        df = df.head(500)

    return _normalise_reddit_df(df, schema_map=None)


def _normalise_reddit_df(
    df: pd.DataFrame,
    schema_map: Optional[dict],
) -> pd.DataFrame:
    """
    Normalise raw Reddit DataFrame to the standard RSSS schema.
    Column names vary between Pushshift dumps, HF datasets, and Kaggle exports.
    """
    # Attempt column mapping from schema_map, then fall back to common aliases
    col_aliases = {
        "post_id": ["id", "post_id", "link_id"],
        "subreddit": ["subreddit", "subreddit_name_prefixed"],
        "title": ["title"],
        "body": ["selftext", "body", "text"],
        "upvotes": ["score", "upvotes", "ups"],
        "comment_count": ["num_comments", "comment_count", "comments"],
        "author": ["author", "author_fullname", "username"],
        "created_utc_unix": ["created_utc", "timestamp_utc", "created"],
    }

    if schema_map:
        explicit = {
            "post_id": schema_map.get("post_id_col"),
            "subreddit": schema_map.get("subreddit_col"),
            "title": schema_map.get("text_col"),
            "body": schema_map.get("body_col"),
            "upvotes": schema_map.get("score_col"),
            "author": schema_map.get("author_col"),
            "created_utc_unix": schema_map.get("ts_col"),
        }
        for std_col, raw_col in explicit.items():
            if raw_col and raw_col in df.columns:
                col_aliases[std_col] = [raw_col] + col_aliases.get(std_col, [])

    result = pd.DataFrame()
    for std_name, aliases in col_aliases.items():
        for alias in aliases:
            if alias and alias in df.columns:
                result[std_name] = df[alias]
                break
        else:
            if std_name not in {"comment_count", "body"}:  # optional
                log.warning("reddit_column_missing", column=std_name, tried=aliases)
                result[std_name] = None

    # Fill optional columns
    if "comment_count" not in result.columns:
        result["comment_count"] = 0
    if "body" not in result.columns:
        result["body"] = ""

    # Convert Unix timestamp to UTC datetime string
    result["created_utc_unix"] = pd.to_numeric(result["created_utc_unix"], errors="coerce")
    result = result.dropna(subset=["created_utc_unix"])
    result["timestamp"] = result["created_utc_unix"].apply(
        lambda ts: unix_to_utc(ts).isoformat()
    )

    # Ensure numeric types
    result["upvotes"] = pd.to_numeric(result.get("upvotes", 0), errors="coerce").fillna(0).astype(int)
    result["comment_count"] = pd.to_numeric(result.get("comment_count", 0), errors="coerce").fillna(0).astype(int)

    # Filter to finance subreddits only
    if "subreddit" in result.columns:
        result["subreddit"] = result["subreddit"].str.lower().str.replace("r/", "", regex=False)
        result = result[result["subreddit"].isin(MONITORED_SUBREDDITS)]
        log.info(
            "reddit_subreddit_filtered",
            rows_kept=len(result),
            subreddits=list(result["subreddit"].unique()),
        )

    result = result.reset_index(drop=True)
    log.info("reddit_df_normalised", rows=len(result), columns=list(result.columns))
    return result


# ---------------------------------------------------------------------------
# Ticker Extraction
# ---------------------------------------------------------------------------

def load_ticker_list() -> set[str]:
    """
    Load the verified ticker list from config/tickers.txt.
    Never hardcode tickers in Python (§17).
    """
    ticker_file = Path(__file__).parent.parent / "config" / "tickers.txt"
    if not ticker_file.exists():
        raise FileNotFoundError(f"Ticker list not found: {ticker_file}")

    tickers = set()
    for line in ticker_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.add(line.upper())
    return tickers


def load_false_positives() -> set[str]:
    """Load false positive tokens from config/false_positive_list.txt."""
    fp_file = Path(__file__).parent.parent / "config" / "false_positive_list.txt"
    if not fp_file.exists():
        return set()

    fps = set()
    for line in fp_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fps.add(line.upper())
    return fps


def extract_ticker_mentions(
    df: pd.DataFrame,
    target_tickers: list[str],
    false_positives: set[str],
) -> pd.DataFrame:
    """
    Extract ticker mentions from post title + body.

    Matches uppercase 1–5 letter tokens against the target_tickers list,
    filtering out false positives (§4.1).

    A post may map to multiple tickers — split it (do NOT deduplicate).

    Args:
        df: Normalised Reddit DataFrame
        target_tickers: Verified ticker symbols to match against
        false_positives: Set of tokens to always reject

    Returns:
        DataFrame with one row per (post_id, ticker) pair.
        Same columns as input plus 'ticker' column.
    """
    import re

    ticker_set = set(t.upper() for t in target_tickers) - false_positives
    token_pattern = re.compile(r"\b[A-Z]{1,5}\b")

    rows = []
    for _, row in df.iterrows():
        text = str(row.get("title", "")) + " " + str(row.get("body", "") or "")
        found_tickers = set()
        for token in token_pattern.findall(text):
            if token in ticker_set and token not in false_positives:
                found_tickers.add(token)

        for ticker in found_tickers:
            new_row = row.to_dict()
            new_row["ticker"] = ticker
            rows.append(new_row)

    result = pd.DataFrame(rows)
    log.info(
        "ticker_mentions_extracted",
        total_posts=len(df),
        total_mentions=len(result),
        unique_tickers=list(result["ticker"].unique()) if len(result) > 0 else [],
    )
    return result


# ---------------------------------------------------------------------------
# FinBERT Sentiment
# ---------------------------------------------------------------------------

def run_finbert(
    texts: list[str],
    batch_size: int = FINBERT_BATCH_SIZE,
    device: str = FINBERT_DEVICE,
) -> list[dict]:
    """
    Run FinBERT on a list of texts and return sentiment scores.

    Falls back to CPU silently if CUDA is unavailable.
    Hard stops if >FINBERT_MAX_FAILURE_RATE of texts fail (§15).

    Args:
        texts: List of text strings to score
        batch_size: Batch size for inference
        device: "cuda", "mps", or "cpu"

    Returns:
        List of dicts: {sentiment_score, sentiment_label, confidence, model_used}
        Failed posts get {sentiment_score: None, confidence: 0.0, reason: "model_failure"}

    Raises:
        RuntimeError: If failure rate exceeds FINBERT_MAX_FAILURE_RATE.
    """
    log.info("finbert_starting", n_texts=len(texts), device=device, batch_size=batch_size)

    try:
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise RuntimeError(
            "transformers/torch not installed. Run: pip install transformers torch"
        ) from e

    # Resolve device — never crash silently on CUDA OOM (§CLAUDE_CODE_INSTRUCTIONS)
    if device == "cuda":
        if not torch.cuda.is_available():
            log.warning("cuda_unavailable_falling_back_to_cpu", requested_device=device)
            device = "cpu"
    elif device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            log.warning("mps_unavailable_falling_back_to_cpu", requested_device=device)
            device = "cpu"

    try:
        pipe = pipeline(
            "text-classification",
            model=FINBERT_MODEL_NAME,
            tokenizer=FINBERT_MODEL_NAME,
            device=0 if device == "cuda" else -1,
            truncation=True,
            max_length=512,
            top_k=None,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load FinBERT model: {e}") from e

    LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    results = []
    n_failed = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            raw_outputs = pipe(batch)
            for output in raw_outputs:
                # output is a list of {label, score} for all classes (top_k=None)
                best = max(output, key=lambda x: x["score"])
                label = best["label"].lower()
                score = LABEL_TO_SCORE.get(label, 0.0)
                results.append({
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "confidence": float(best["score"]),
                    "model_used": "finbert",
                })
        except Exception as e:
            log.error("finbert_batch_failed", batch_start=i, error=str(e))
            for _ in batch:
                results.append({
                    "sentiment_score": None,
                    "confidence": 0.0,
                    "model_used": "finbert",
                    "reason": "model_failure",
                })
                n_failed += 1

    failure_rate = n_failed / max(len(texts), 1)
    if failure_rate > FINBERT_MAX_FAILURE_RATE:
        raise RuntimeError(
            f"FinBERT failure rate {failure_rate:.1%} exceeds threshold "
            f"{FINBERT_MAX_FAILURE_RATE:.1%}. Hard stop per §15."
        )

    log.info(
        "finbert_complete",
        n_texts=len(texts),
        n_failed=n_failed,
        failure_rate=round(failure_rate, 4),
    )
    return results


def add_sentiment_scores(mentions_df: pd.DataFrame, skip_finbert: bool = False) -> pd.DataFrame:
    """
    Add FinBERT sentiment columns to the mentions DataFrame.

    If skip_finbert=True, sentiment columns are set to NaN (use for fast baseline IC check).

    Args:
        mentions_df: DataFrame with 'title' and 'body' columns
        skip_finbert: If True, skip model inference

    Returns:
        DataFrame with added columns: sentiment_score, sentiment_label,
        sentiment_confidence, model_used.
    """
    if skip_finbert:
        log.warning("finbert_skipped_sentiment_null_for_all_rows")
        mentions_df = mentions_df.copy()
        mentions_df["sentiment_score"] = np.nan
        mentions_df["sentiment_label"] = None
        mentions_df["sentiment_confidence"] = np.nan
        mentions_df["model_used"] = None
        return mentions_df

    texts = (
        (mentions_df["title"].fillna("") + " " + mentions_df["body"].fillna(""))
        .str.strip()
        .tolist()
    )

    scores = run_finbert(texts)
    score_df = pd.DataFrame(scores)
    return pd.concat(
        [mentions_df.reset_index(drop=True), score_df.reset_index(drop=True)],
        axis=1,
    )


# ---------------------------------------------------------------------------
# IC Computation
# ---------------------------------------------------------------------------

def compute_daily_sentiment_features(
    mentions_df: pd.DataFrame,
    ticker: str,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Aggregate per-post sentiment into daily features for a single ticker.

    CRITICAL: Uses strict < on timestamp cutoffs to prevent leakage.
    This is a simplified version of features/reddit_features.py for Phase 0.

    Args:
        mentions_df: Ticker-filtered DataFrame with 'timestamp' and 'sentiment_score'
        ticker: Ticker symbol
        trading_days: NYSE trading days in the analysis window

    Returns:
        DataFrame with columns: date, ticker, post_count_1d, post_count_3d,
        avg_sentiment_1d, avg_sentiment_3d, sentiment_acceleration, mention_growth
    """
    ticker_df = mentions_df[mentions_df["ticker"] == ticker].copy()
    ticker_df["ts_utc"] = pd.to_datetime(ticker_df["timestamp"], utc=True)

    rows = []
    for ts in trading_days:
        row_date = ts.date()
        cutoff = market_open_utc(row_date)
        cutoff_ts = pd.Timestamp(cutoff)

        window_1d_start = cutoff_ts - pd.Timedelta(hours=24)
        window_3d_start = cutoff_ts - pd.Timedelta(hours=72)

        # STRICT less-than on upper bound (§3.1)
        posts_1d = ticker_df[
            (ticker_df["ts_utc"] >= window_1d_start)
            & (ticker_df["ts_utc"] < cutoff_ts)
        ]
        posts_3d = ticker_df[
            (ticker_df["ts_utc"] >= window_3d_start)
            & (ticker_df["ts_utc"] < cutoff_ts)
        ]

        post_count_1d = len(posts_1d)
        post_count_3d = len(posts_3d)

        # Quality gate
        if post_count_1d < MIN_POST_COUNT_1D:
            continue

        valid_1d = posts_1d.dropna(subset=["sentiment_score"])
        valid_3d = posts_3d.dropna(subset=["sentiment_score"])

        avg_sentiment_1d = float(valid_1d["sentiment_score"].mean()) if not valid_1d.empty else None
        avg_sentiment_3d = float(valid_3d["sentiment_score"].mean()) if not valid_3d.empty else None

        sentiment_acceleration = None
        if avg_sentiment_1d is not None and avg_sentiment_3d is not None:
            sentiment_acceleration = avg_sentiment_1d - avg_sentiment_3d

        mention_growth = post_count_1d / (post_count_3d + 1.0)

        rows.append({
            "date": row_date,
            "ticker": ticker,
            "post_count_1d": post_count_1d,
            "post_count_3d": post_count_3d,
            "avg_sentiment_1d": avg_sentiment_1d,
            "avg_sentiment_3d": avg_sentiment_3d,
            "sentiment_acceleration": sentiment_acceleration,
            "mention_growth": mention_growth,
        })

    return pd.DataFrame(rows)


def compute_ic_for_ticker(
    sentiment_features: pd.DataFrame,
    market_df: pd.DataFrame,
    ticker: str,
) -> dict:
    """
    Compute Spearman IC between sentiment features and forward returns.

    Args:
        sentiment_features: Daily sentiment features for this ticker
        market_df: OHLCV DataFrame for this ticker
        ticker: For logging

    Returns:
        Dict with IC values for each feature × horizon combination.
    """
    if sentiment_features.empty:
        log.warning("ic_no_sentiment_data", ticker=ticker)
        return {}

    market_dates = pd.to_datetime(market_df.index).normalize()
    close = market_df["close"].values.astype(float)

    rows_with_labels = []
    for _, row in sentiment_features.iterrows():
        row_date = pd.Timestamp(row["date"])
        pos_matches = (market_dates == row_date).values.nonzero()[0]
        if len(pos_matches) == 0:
            continue
        idx = int(pos_matches[0])
        base_close = close[idx]

        label_row = row.to_dict()
        for horizon, offset in [("return_1d", 1), ("return_5d", 5)]:
            future_idx = idx + offset
            if future_idx < len(close):
                label_row[horizon] = (close[future_idx] - base_close) / base_close
            else:
                label_row[horizon] = None
        rows_with_labels.append(label_row)

    if not rows_with_labels:
        return {}

    combined = pd.DataFrame(rows_with_labels)

    features_to_test = ["sentiment_acceleration", "mention_growth", "avg_sentiment_1d"]
    horizons = ["return_1d", "return_5d"]

    ic_results: dict = {}
    for feat in features_to_test:
        if feat not in combined.columns:
            continue
        for horizon in horizons:
            key = f"ic_{feat}_vs_{horizon}"
            subset = combined[[feat, horizon]].dropna()
            if len(subset) < 10:
                log.warning(
                    "ic_insufficient_samples",
                    ticker=ticker,
                    feature=feat,
                    horizon=horizon,
                    n=len(subset),
                )
                ic_results[key] = None
                continue
            ic_val, p_val = spearmanr(subset[feat], subset[horizon])
            ic_results[key] = float(ic_val) if not np.isnan(ic_val) else None
            log.info(
                "ic_computed",
                ticker=ticker,
                feature=feat,
                horizon=horizon,
                ic=round(float(ic_val), 4) if not np.isnan(ic_val) else None,
                p_value=round(float(p_val), 4),
                n_samples=len(subset),
            )

    ic_results["sample_size"] = len(combined)
    return ic_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Phase 0 IC validation and write ic_report.json."""
    args = parse_args()

    if args.list_datasets:
        print("\nKnown HuggingFace dataset candidates (ranked by fit):")
        for i, ds in enumerate(PHASE0_HF_DATASETS, 1):
            print(f"\n  {i}. {ds['id']}")
            print(f"     {ds['notes']}")
            print(f"     Load: from datasets import load_dataset; ds = load_dataset('{ds['id']}')")
        sys.exit(0)

    log.info(
        "phase0_starting",
        dataset=args.dataset,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        debug=args.debug,
        skip_finbert=args.skip_finbert,
    )

    # --- 1. Load Reddit dataset ---
    try:
        if args.dataset == "local":
            if not args.local_path:
                log.error("local_path_required_when_dataset_is_local")
                sys.exit(1)
            reddit_df = load_local_dataset(args.local_path, debug=args.debug)
        else:
            reddit_df = load_hf_dataset(args.dataset, debug=args.debug)
    except Exception as e:
        log.error("reddit_dataset_load_failed", error=str(e))
        print(f"\nERROR: Failed to load dataset '{args.dataset}': {e}", file=sys.stderr)
        print(
            "\nTry one of these alternatives:\n"
            "  --dataset RomanBlanco/reddit_wsb_2021\n"
            "  --dataset SocialGrep/one-million-reddit-comments\n"
            "  --dataset local --local-path data/raw/wsb_posts.csv\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 2. Extract ticker mentions ---
    false_positives = load_false_positives()
    mentions_df = extract_ticker_mentions(reddit_df, args.tickers, false_positives)

    if mentions_df.empty:
        log.error("no_ticker_mentions_found", tickers=args.tickers)
        print(
            "\nERROR: No ticker mentions found for any Phase 0 tickers.\n"
            "The dataset may not contain finance-relevant posts for these symbols.\n"
            "Check that subreddit filtering is not too aggressive and the dataset "
            "covers the target tickers.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 3. Run FinBERT ---
    mentions_df = add_sentiment_scores(mentions_df, skip_finbert=args.skip_finbert)

    # --- 4. Load market data ---
    market_data = load_ohlcv_batch(
        args.tickers,
        start_date=args.start_date,
        end_date=(
            pd.Timestamp(args.end_date) + pd.Timedelta(days=10)
        ).strftime("%Y-%m-%d"),  # extra days for forward return computation
    )

    if not market_data:
        log.error("market_data_completely_unavailable")
        sys.exit(1)

    # --- 5. Compute IC per ticker ---
    trading_days = get_trading_days(args.start_date, args.end_date)
    ticker_results: dict = {}

    for ticker in args.tickers:
        if ticker not in market_data:
            log.warning("ticker_skipped_no_market_data", ticker=ticker)
            continue

        log.info("processing_ticker", ticker=ticker)
        sentiment_features = compute_daily_sentiment_features(
            mentions_df, ticker, trading_days
        )
        ic_values = compute_ic_for_ticker(
            sentiment_features, market_data[ticker], ticker
        )

        if not ic_values:
            ticker_results[ticker] = {
                "verdict": "NO_DATA",
                "reason": "insufficient_sentiment_or_market_data",
            }
            continue

        # Collect key IC values
        ic_accel_5d = ic_values.get("ic_sentiment_acceleration_vs_return_5d")
        ic_accel_1d = ic_values.get("ic_sentiment_acceleration_vs_return_1d")
        ic_growth_1d = ic_values.get("ic_mention_growth_vs_return_1d")

        # Verdict: SIGNAL_DETECTED if any key IC exceeds threshold
        ic_values_numeric = [
            v for v in [ic_accel_5d, ic_accel_1d, ic_growth_1d]
            if v is not None and not np.isnan(v)
        ]
        max_ic = max(abs(v) for v in ic_values_numeric) if ic_values_numeric else 0.0
        verdict = "SIGNAL_DETECTED" if max_ic >= IC_ABORT_THRESHOLD else "NO_SIGNAL"

        ticker_results[ticker] = {
            **ic_values,
            "max_abs_ic": round(max_ic, 4),
            "verdict": verdict,
        }
        log.info(
            "ticker_verdict",
            ticker=ticker,
            max_abs_ic=round(max_ic, 4),
            verdict=verdict,
        )

    # --- 6. Overall verdict ---
    ic_by_ticker = [
        abs(r.get("max_abs_ic", 0.0))
        for r in ticker_results.values()
        if isinstance(r.get("max_abs_ic"), float)
    ]
    median_ic = float(np.median(ic_by_ticker)) if ic_by_ticker else 0.0
    overall_verdict = "PROCEED" if median_ic >= IC_ABORT_THRESHOLD else "ABORT"

    report = {
        "run_date": date.today().isoformat(),
        "dataset": args.dataset,
        "tickers_tested": args.tickers,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "debug_mode": args.debug,
        "skip_finbert": args.skip_finbert,
        "results": ticker_results,
        "overall_verdict": overall_verdict,
        "median_ic": round(median_ic, 4),
        "threshold_used": IC_ABORT_THRESHOLD,
    }

    # --- 7. Write results ---
    output_path = PHASE0_RESULTS_PATH / "ic_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(
        "phase0_complete",
        overall_verdict=overall_verdict,
        median_ic=round(median_ic, 4),
        threshold=IC_ABORT_THRESHOLD,
        output_path=str(output_path),
    )

    # --- 8. Print summary ---
    print("\n" + "=" * 60)
    print(f"  Phase 0 — Signal Validation Result: {overall_verdict}")
    print("=" * 60)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Tickers:     {', '.join(args.tickers)}")
    print(f"  Median IC:   {median_ic:.4f}  (threshold: {IC_ABORT_THRESHOLD})")
    print()
    for ticker, result in ticker_results.items():
        verdict = result.get("verdict", "?")
        ic = result.get("max_abs_ic", "n/a")
        print(f"  {ticker:6s}  IC={ic!s:8s}  {verdict}")
    print()
    if overall_verdict == "PROCEED":
        print("  ✓ Signal detected. Proceed to Phase 1 — Data Pipeline.")
    else:
        print("  ✗ IC below threshold. ABORT — reassess thesis before building infrastructure.")
    print(f"\n  Full report: {output_path}")
    print("=" * 60)

    sys.exit(0 if overall_verdict == "PROCEED" else 1)


if __name__ == "__main__":
    main()
