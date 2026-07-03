"""
backfill_gdelt_news.py
======================
Download and process GDELT GKG v1 daily files for 2023-01-01 to 2024-12-31.

Uses GKG v1 bulk files (no API key, free public dataset).
Filters to finance articles, extracts tickers via company-name aliases,
aggregates to one daily row per ticker.

Output:   data/processed/gdelt_news_features.parquet
Progress: data/processed/gdelt_progress.json

Usage:
    python scripts/backfill_gdelt_news.py --yes           # full backfill
    python scripts/backfill_gdelt_news.py --test          # 2024-01-15 only
    python scripts/backfill_gdelt_news.py --resume --yes  # resume
    python scripts/backfill_gdelt_news.py --status        # show progress
    python scripts/backfill_gdelt_news.py --start 2024-01-01 --end 2024-06-30 --yes
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_START = "2023-01-01"
DEFAULT_END   = "2024-12-31"
DOWNLOAD_SLEEP = 2.0   # seconds between file downloads (polite)
SAVE_EVERY     = 30    # checkpoint every N days

OUTPUT_PATH   = "data/processed/gdelt_news_features.parquet"
PROGRESS_PATH = "data/processed/gdelt_progress.json"


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if Path(PROGRESS_PATH).exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed_dates": [], "total_records": 0, "last_updated": None}


def save_progress(progress: dict) -> None:
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    Path(PROGRESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet append
# ─────────────────────────────────────────────────────────────────────────────

def append_to_parquet(rows: list[dict]) -> None:
    """Append rows, dedup on (ticker, date). Never overwrites existing data."""
    if not rows:
        return

    df_new = pd.DataFrame(rows)

    if Path(OUTPUT_PATH).exists():
        df_old = pd.read_parquet(OUTPUT_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Date iteration helper
# ─────────────────────────────────────────────────────────────────────────────

def all_days(start: str, end: str) -> list[str]:
    """Return list of YYYYMMDD strings for every calendar day in range."""
    days   = []
    cur    = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end,   "%Y-%m-%d")
    while cur <= end_dt:
        days.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return days


# ─────────────────────────────────────────────────────────────────────────────
# Estimate
# ─────────────────────────────────────────────────────────────────────────────

def print_estimate(days: list[str], yes_flag: bool) -> None:
    est_min = len(days) * DOWNLOAD_SLEEP / 60

    print()
    print("=" * 58)
    print("GDELT Backfill Estimate:")
    print(f"  Date range:        {days[0][:4]}-{days[0][4:6]}-{days[0][6:]} "
          f"→ {days[-1][:4]}-{days[-1][4:6]}-{days[-1][6:]}")
    print(f"  Total days:        {len(days)}")
    print(f"  Download delay:    {DOWNLOAD_SLEEP}s per file")
    print(f"  Estimated time:    ~{est_min:.0f} minutes")
    print(f"  File size:         ~30–50MB per day (compressed)")
    print("  Please run during off-peak hours — be a good citizen.")
    print("=" * 58)
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
    progress  = load_progress()
    completed = progress.get("completed_dates", [])

    print()
    print("=" * 48)
    print("GDELT BACKFILL STATUS")
    print("=" * 48)
    print(f"Completed dates:   {len(completed):,}")
    print(f"Total records:     {progress.get('total_records', 0):,}")
    print(f"Last updated:      {progress.get('last_updated', 'never')}")
    if completed:
        sorted_d = sorted(completed)
        print(f"First done:        {sorted_d[0]}")
        print(f"Last done:         {sorted_d[-1]}")
    if Path(OUTPUT_PATH).exists():
        size_mb = Path(OUTPUT_PATH).stat().st_size / (1024 ** 2)
        df      = pd.read_parquet(OUTPUT_PATH)
        print(f"Output file:       {OUTPUT_PATH} ({size_mb:.1f}MB)")
        print(f"Output rows:       {len(df):,}")
        print(f"Tickers covered:   {df['ticker'].nunique()}")
        print(f"Date range:        {df['date'].min()} → {df['date'].max()}")
    else:
        print("Output file:       not yet created")
    print("=" * 48)


# ─────────────────────────────────────────────────────────────────────────────
# Test mode
# ─────────────────────────────────────────────────────────────────────────────

def run_test() -> None:
    """Process a single day (2024-01-15) and print results without saving."""
    from data.gdelt_news_fetcher import download_gkg_day, compute_gdelt_daily_sentiment

    test_date = "20240115"
    logger.info(f"TEST MODE — processing {test_date}")

    df = download_gkg_day(test_date)
    if df.empty:
        print("ERROR: Failed to download GKG file. Check connectivity.")
        return

    print(f"\nDownloaded: {test_date}.gkg.csv ({len(df):,} rows)")

    rows = compute_gdelt_daily_sentiment(df, test_date)

    if not rows:
        print("No finance articles with ticker matches found.")
        return

    # Count finance rows (approximate: rows that passed finance filter)
    from data.gdelt_news_fetcher import is_finance_article
    fin_rows = df.apply(is_finance_article, axis=1).sum()
    print(f"Finance articles filtered: {fin_rows:,}")
    print(f"Tickers found: {len(rows)}")
    print()
    print("Tickers found:")
    for r in sorted(rows, key=lambda x: -x["gdelt_article_count"]):
        print(
            f"  {r['ticker']:<6}  {r['gdelt_article_count']:>3} articles  "
            f"tone={r['gdelt_tone_mean']:+.3f}"
        )
    print(f"\nTotal records: {len(rows)} rows")
    print("NOTE: Test mode — no file saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main backfill loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backfill(
    start: str,
    end: str,
    resume: bool = False,
) -> None:
    from data.gdelt_news_fetcher import download_gkg_day, compute_gdelt_daily_sentiment, is_finance_article

    progress  = load_progress() if resume else {
        "completed_dates": [],
        "total_records":   0,
        "last_updated":    None,
    }
    completed = set(progress.get("completed_dates", []))
    days      = all_days(start, end)

    logger.info(
        f"Starting GDELT backfill: {len(days)} days, "
        f"{start} → {end}"
    )

    batch_rows: list[dict] = []
    start_time = time.time()

    for i, date_str in enumerate(days, start=1):
        if date_str in completed:
            continue

        # Download
        df   = download_gkg_day(date_str)
        rows = []

        if not df.empty:
            rows = compute_gdelt_daily_sentiment(df, date_str)

        batch_rows.extend(rows)

        completed.add(date_str)
        progress["completed_dates"] = list(completed)
        progress["total_records"]   = (
            progress.get("total_records", 0) + len(rows)
        )

        # Progress report every 10 days
        if i % 10 == 0:
            elapsed  = time.time() - start_time
            rate     = i / elapsed if elapsed > 0 else 0.001
            eta_sec  = (len(days) - i) / rate
            eta_m    = int(eta_sec // 60)
            fin_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            fin_count = (
                int(df.apply(is_finance_article, axis=1).sum())
                if not df.empty else 0
            )
            logger.info(
                f"{fin_date} | finance_rows={fin_count} | "
                f"tickers_found={len(rows)} | "
                f"total={progress['total_records']:,} | "
                f"eta={eta_m}m"
            )

        # Checkpoint save every SAVE_EVERY days
        if len(completed) % SAVE_EVERY == 0:
            save_progress(progress)
            if batch_rows:
                append_to_parquet(batch_rows)
                logger.info(
                    f"Checkpoint: saved {len(batch_rows)} rows"
                )
                batch_rows = []

        time.sleep(DOWNLOAD_SLEEP)

    # Final save
    save_progress(progress)
    if batch_rows:
        append_to_parquet(batch_rows)

    # Summary
    print()
    print("=" * 42)
    print("GDELT BACKFILL COMPLETE")
    print("=" * 42)
    start_fmt = f"{start[:4]}-{start[5:7]}-{start[8:10]}"
    end_fmt   = f"{end[:4]}-{end[5:7]}-{end[8:10]}"
    print(f"Date range:      {start_fmt} → {end_fmt}")
    print(f"Total records:   {progress['total_records']:,}")

    if Path(OUTPUT_PATH).exists():
        df      = pd.read_parquet(OUTPUT_PATH)
        size_mb = Path(OUTPUT_PATH).stat().st_size / (1024 ** 2)
        print(f"Tickers covered: {df['ticker'].nunique()}")
        print(f"Output:          {OUTPUT_PATH}")
        print(f"Size:            ~{size_mb:.0f}MB")
    print("=" * 42)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GDELT GKG v1 news backfill — 2023-2024",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backfill_gdelt_news.py --test
  python scripts/backfill_gdelt_news.py --yes
  python scripts/backfill_gdelt_news.py --resume --yes
  python scripts/backfill_gdelt_news.py --status
  python scripts/backfill_gdelt_news.py --start 2024-01-01 --end 2024-06-30 --yes
        """,
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only 2024-01-15 and print results — no file saved",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
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
    args = parse_args()

    if args.status:
        print_status()
        return

    if args.test:
        run_test()
        return

    days = all_days(args.start, args.end)
    if not args.resume:
        print_estimate(days, args.yes)

    run_backfill(
        start  = args.start,
        end    = args.end,
        resume = args.resume,
    )


if __name__ == "__main__":
    main()
