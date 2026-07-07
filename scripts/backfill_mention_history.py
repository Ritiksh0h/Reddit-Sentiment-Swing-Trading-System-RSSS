#!/usr/bin/env python3
"""
Backfill data/mention_history.json with historical Reddit post counts.

Fetches daily post counts for each tracked ticker from Arctic Shift for a
specified date range, then merges into the existing mention_history.json so
abnormal_attention_1d gets a real 20-day rolling baseline.

Counts replicate the live pipeline exactly (apples-to-apples baseline):
  - same 48h-wide delayed window shape as fetch_recent_posts (72h -> 24h ago),
    anchored at 14:00 ET on the target date (the last daily run of the day,
    whose write wins in mention_history)
  - same extractor (extract_tickers_from_text, titles only)
  - same 5-page-per-subreddit truncation

Usage:
    python3 scripts/backfill_mention_history.py
    python3 scripts/backfill_mention_history.py --start 2026-06-07 --end 2026-06-16
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
import argparse
from datetime import datetime, date, timezone, timedelta
from collections import defaultdict
from pathlib import Path

from data.reddit_live_fetcher import (
    _fetch_subreddit_page,
    extract_tickers_from_text,
    TRACKED_SUBREDDITS,
    TRACKED_TICKERS,
    API_PAGE_LIMIT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

HISTORY_PATH = Path('data/mention_history.json')
MAX_PAGES    = 5   # same truncation as live fetch_recent_posts


def _fetch_page_with_retry(subreddit: str, after: int, before: int) -> list:
    """One retry — Arctic Shift has been intermittently 500-ing."""
    try:
        return _fetch_subreddit_page(subreddit, after, before)
    except Exception as e:
        logger.warning(f'page_fetch_retry subreddit={subreddit}: {e}')
        time.sleep(2.0)
        return _fetch_subreddit_page(subreddit, after, before)


def fetch_counts_for_date(target_date: date) -> dict | None:
    """
    Fetch post counts per ticker for one date, using the same window shape the
    live pipeline stores under that date key. Returns {ticker: count}, or None
    when the day's data is unusable (total API failure / wallstreetbets down —
    a missing date beats a falsely-low baseline entry).
    """
    # Live: after = now-72h, before = now-24h, with now ~ 14:00 ET (18:00 UTC)
    anchor = datetime(target_date.year, target_date.month, target_date.day,
                      18, 0, 0, tzinfo=timezone.utc)
    after  = int((anchor - timedelta(hours=72)).timestamp())
    before = int((anchor - timedelta(hours=24)).timestamp())

    ticker_counts: dict = defaultdict(int)
    total_posts = 0
    failed_subs = []

    for subreddit in TRACKED_SUBREDDITS:
        page_before = before
        try:
            for _ in range(MAX_PAGES):
                posts = _fetch_page_with_retry(subreddit, after, page_before)
                if not posts:
                    break
                for post in posts:
                    title = post.get('title', '') or ''
                    for ticker in extract_tickers_from_text(title):
                        ticker_counts[ticker] += 1
                total_posts += len(posts)
                if len(posts) < API_PAGE_LIMIT:
                    break
                oldest = min(p.get('created_utc', before) for p in posts)
                page_before = oldest - 1
                if page_before <= after:
                    break
                time.sleep(0.3)
            time.sleep(0.5)  # polite between subreddits
        except Exception as e:
            logger.warning(f'fetch_failed date={target_date} subreddit={subreddit}: {e}')
            failed_subs.append(subreddit)

    if total_posts == 0 or 'wallstreetbets' in failed_subs:
        logger.error(f'date={target_date} unusable (total={total_posts} '
                     f'failed={failed_subs}) — day skipped, not written')
        return None

    logger.info(f'date={target_date} total_posts={total_posts} '
                f'tickers_found={len(ticker_counts)} failed_subs={failed_subs}')
    return dict(ticker_counts)


def load_history() -> dict:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return {}


def save_history(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, sort_keys=True))


def backfill(start_date: date, end_date: date) -> None:
    history = load_history()
    all_days = [start_date + timedelta(days=i)
                for i in range((end_date - start_date).days + 1)]

    # Weekdays only — live runs Mon-Fri, so history keys are weekdays
    trading_days = [d for d in all_days if d.weekday() < 5]
    logger.info(f'Backfilling {len(trading_days)} days: '
                f'{trading_days[0]} -> {trading_days[-1]}')

    for day in trading_days:
        day_str = day.isoformat()

        # Idempotence spot check — skip days already backfilled
        already_have = all(
            day_str in history.get(ticker, {})
            for ticker in TRACKED_TICKERS[:5]
        )
        if already_have:
            logger.info(f'date={day_str} already in history — skipping')
            continue

        counts = fetch_counts_for_date(day)
        if counts is None:
            continue

        # Zero for tracked tickers with no mentions — quiet days must drag
        # the rolling average down, same as the training distribution
        for ticker in TRACKED_TICKERS:
            if ticker not in history:
                history[ticker] = {}
            history[ticker][day_str] = counts.get(ticker, 0)

        save_history(history)
        logger.info(f'date={day_str} saved — '
                    f'{sum(1 for c in counts.values() if c > 0)} tickers with mentions')
        time.sleep(1.0)  # polite between days

    logger.info('Backfill complete')

    min_days = min(len(v) for v in history.values() if v)
    max_days = max(len(v) for v in history.values() if v)
    over20   = [t for t, v in history.items() if len(v) >= 20]
    logger.info(f'History after backfill: min={min_days} max={max_days} '
                f'tickers_over20={len(over20)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2026-06-07',
                        help='Start date YYYY-MM-DD (default: 2026-06-07)')
    parser.add_argument('--end', default='2026-06-16',
                        help='End date YYYY-MM-DD (default: 2026-06-16)')
    args = parser.parse_args()

    backfill(date.fromisoformat(args.start), date.fromisoformat(args.end))
