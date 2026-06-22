"""
backfill_finnhub_news.py
========================
Backfill last ~12 months of Finnhub company-news for all 38 tickers.
(Finnhub free tier covers ~12 months of history.)

Output:   data/processed/finnhub_news_features.parquet
Progress: data/processed/finnhub_progress.json

Resume-safe: completed tickers are skipped on restart.

Usage:
    python scripts/backfill_finnhub_news.py --yes        # skip prompt
    python scripts/backfill_finnhub_news.py --resume     # resume
    python scripts/backfill_finnhub_news.py --status     # show progress
    python scripts/backfill_finnhub_news.py --ticker NVDA  # single ticker
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH   = "data/processed/finnhub_news_features.parquet"
PROGRESS_PATH = "data/processed/finnhub_progress.json"

TODAY      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
START_DATE = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if Path(PROGRESS_PATH).exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed_tickers": [], "total_articles": 0, "last_updated": None}


def save_progress(progress: dict) -> None:
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    Path(PROGRESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def append_to_parquet(rows: list[dict]) -> None:
    """Append new rows, dedup on (ticker, date), never overwrite."""
    if not rows:
        return

    df_new = pd.DataFrame(rows)

    if Path(OUTPUT_PATH).exists():
        df_old = pd.read_parquet(OUTPUT_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Estimate
# ─────────────────────────────────────────────────────────────────────────────

def print_estimate(tickers: list[str], yes_flag: bool) -> None:
    n_batches = len(tickers) * 12  # ~12 monthly calls per ticker
    est_min   = n_batches * 1.1 / 60

    print()
    print("=" * 52)
    print("Finnhub Backfill Estimate:")
    print(f"  Tickers:           {len(tickers)}")
    print(f"  Date range:        {START_DATE} → {TODAY}")
    print(f"  API calls:         ~{n_batches} (monthly batches)")
    print(f"  Estimated time:    ~{est_min:.0f} minutes")
    print(f"  Rate limit:        1.1s between calls")
    print("=" * 52)
    print()

    if yes_flag:
        return
    try:
        input("Press ENTER to continue or Ctrl+C to cancel... ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

def print_status() -> None:
    progress = load_progress()
    completed = progress.get("completed_tickers", [])

    print()
    print("=" * 45)
    print("FINNHUB BACKFILL STATUS")
    print("=" * 45)
    print(f"Completed tickers: {len(completed)}")
    print(f"Total articles:    {progress.get('total_articles', 0):,}")
    print(f"Last updated:      {progress.get('last_updated', 'never')}")
    if completed:
        print(f"Tickers done:      {', '.join(sorted(completed))}")
    if Path(OUTPUT_PATH).exists():
        size_mb = Path(OUTPUT_PATH).stat().st_size / (1024 ** 2)
        df = pd.read_parquet(OUTPUT_PATH)
        print(f"Output file:       {OUTPUT_PATH} ({size_mb:.1f}MB)")
        print(f"Output rows:       {len(df):,}")
        print(f"Date range:        {df['date'].min()} → {df['date'].max()}")
    else:
        print("Output file:       not yet created")
    print("=" * 45)


# ─────────────────────────────────────────────────────────────────────────────
# Main backfill
# ─────────────────────────────────────────────────────────────────────────────

def run_backfill(
    tickers: list[str],
    api_key: str,
    start: str = START_DATE,
    end: str   = TODAY,
    resume: bool = False,
) -> None:
    from data.finnhub_news_fetcher import fetch_ticker_history, articles_to_daily_rows

    progress  = load_progress() if resume else {
        "completed_tickers": [],
        "total_articles": 0,
        "last_updated": None,
    }
    completed = set(progress.get("completed_tickers", []))

    logger.info(
        f"Starting Finnhub backfill: {len(tickers)} tickers, "
        f"{start} → {end}"
    )

    for i, ticker in enumerate(tickers, start=1):
        if ticker in completed:
            logger.info(f"[{i}/{len(tickers)}] {ticker} — already complete, skipping")
            continue

        logger.info(f"[{i}/{len(tickers)}] Fetching {ticker} {start} → {end}")

        articles = fetch_ticker_history(ticker, start, end, api_key)
        rows     = articles_to_daily_rows(articles, ticker)

        if rows:
            append_to_parquet(rows)
            logger.info(
                f"  {ticker}: {len(articles)} articles → "
                f"{len(rows)} daily rows"
            )
        else:
            logger.info(f"  {ticker}: 0 articles in range")

        completed.add(ticker)
        progress["completed_tickers"] = list(completed)
        progress["total_articles"]    = (
            progress.get("total_articles", 0) + len(articles)
        )
        save_progress(progress)

    # Summary
    if Path(OUTPUT_PATH).exists():
        df      = pd.read_parquet(OUTPUT_PATH)
        size_mb = Path(OUTPUT_PATH).stat().st_size / (1024 ** 2)
        print()
        print("=" * 42)
        print("FINNHUB BACKFILL COMPLETE")
        print("=" * 42)
        print(f"Tickers processed: {len(completed)}")
        print(f"Total articles:    {progress['total_articles']:,}")
        print(
            f"Date range:        "
            f"{df['date'].min()} → {df['date'].max()}"
        )
        print(f"Output:            {OUTPUT_PATH}")
        print(f"Size:              ~{size_mb:.0f}MB")
        print(f"Rows:              {len(df):,}")
        print("=" * 42)
    else:
        logger.warning("No data written — all tickers returned 0 articles.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finnhub historical news backfill (~12 months)"
    )
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"Start date YYYY-MM-DD (default: {START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=TODAY,
        help=f"End date YYYY-MM-DD (default: today = {TODAY})",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Backfill a single ticker only",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed tickers",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show progress and exit",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    api_key = os.getenv("FINNHUB_API_KEY")

    if args.status:
        print_status()
        return

    if not api_key:
        logger.error("FINNHUB_API_KEY not set in .env — aborting")
        sys.exit(1)

    from config.settings import load_tickers, TICKERS_TRADE_PATH, TICKERS_WATCH_PATH

    all_tickers = sorted(
        set(load_tickers(TICKERS_TRADE_PATH)) |
        set(load_tickers(TICKERS_WATCH_PATH))
    )
    tickers = [args.ticker.upper()] if args.ticker else all_tickers

    if not args.resume:
        print_estimate(tickers, args.yes)

    run_backfill(
        tickers = tickers,
        api_key = api_key,
        start   = args.start,
        end     = args.end,
        resume  = args.resume,
    )


if __name__ == "__main__":
    main()
